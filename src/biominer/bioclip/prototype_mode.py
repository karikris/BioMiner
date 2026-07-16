"""Fail-closed configuration for the Build Week target-aware prototype mode."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any

from biominer.bioclip.classification_modes import (
    BUILD_WEEK_TARGET_AWARE_PROTOTYPE,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint


BUILD_WEEK_PROTOTYPE_CONFIG_VERSION = "build-week-target-aware-prototype-config-v1.0.0"
BUILD_WEEK_PROTOTYPE_DEPLOYMENT_STATUS = "prototype"
BUILD_WEEK_PROTOTYPE_VISUAL_INPUT = "raw_full_image"


@dataclass(frozen=True, slots=True)
class BuildWeekPrototypeConfig:
    target_accepted_taxon_key: str
    target_scientific_name: str
    reference_bank_readiness: Path
    reference_bank_readiness_sha256: str
    support_manifest: Path
    support_manifest_sha256: str
    reference_embeddings: Path
    reference_embeddings_sha256: str
    candidate_score_evidence: Path
    candidate_score_evidence_sha256: str
    prototype_policy: Path
    prototype_policy_sha256: str
    model_revision: str
    preprocessing_version: str
    visual_input_version: str
    classifier_fingerprint: str
    margin_policy_version: str
    limitations: tuple[str, ...]
    classification_mode: str = BUILD_WEEK_TARGET_AWARE_PROTOTYPE
    deployment_status: str = BUILD_WEEK_PROTOTYPE_DEPLOYMENT_STATUS
    workflow: str = "reference-first"
    storage_backend: str = "local"
    s3_permitted: bool = False
    target_always_scored: bool = True
    complete_regional_candidate_union_scored: bool = True
    hierarchy_pruning_permitted: bool = False
    spatial_crop_permitted: bool = False
    visual_input: str = BUILD_WEEK_PROTOTYPE_VISUAL_INPUT
    prototype_readiness_required: bool = True
    prototype_support_bank_required: bool = True
    silent_fallback_permitted: bool = False
    output_status: str = BUILD_WEEK_PROTOTYPE_DEPLOYMENT_STATUS
    legacy_classifier_available: bool = True
    b0_baseline_available: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "reference_bank_readiness",
            "support_manifest",
            "reference_embeddings",
            "candidate_score_evidence",
            "prototype_policy",
        ):
            raw_path = getattr(self, field_name)
            if "://" in str(raw_path):
                raise ValueError(f"{field_name} must be a local path")
            path = Path(raw_path).expanduser()
            object.__setattr__(self, field_name, path)
        for field_name in (
            "reference_bank_readiness_sha256",
            "support_manifest_sha256",
            "reference_embeddings_sha256",
            "candidate_score_evidence_sha256",
            "prototype_policy_sha256",
            "classifier_fingerprint",
        ):
            _require_sha256(getattr(self, field_name), field=field_name)
        for field_name in (
            "target_accepted_taxon_key",
            "target_scientific_name",
            "model_revision",
            "preprocessing_version",
            "visual_input_version",
            "margin_policy_version",
        ):
            _required_text(getattr(self, field_name), field=field_name)
        limitations = tuple(
            _required_text(value, field="limitations[]") for value in self.limitations
        )
        if not limitations:
            raise ValueError("limitations must not be empty")
        if len(limitations) != len(set(limitations)):
            raise ValueError("limitations must be unique")
        object.__setattr__(self, "limitations", limitations)
        expected = {
            "classification_mode": BUILD_WEEK_TARGET_AWARE_PROTOTYPE,
            "deployment_status": BUILD_WEEK_PROTOTYPE_DEPLOYMENT_STATUS,
            "workflow": "reference-first",
            "storage_backend": "local",
            "s3_permitted": False,
            "target_always_scored": True,
            "complete_regional_candidate_union_scored": True,
            "hierarchy_pruning_permitted": False,
            "spatial_crop_permitted": False,
            "visual_input": BUILD_WEEK_PROTOTYPE_VISUAL_INPUT,
            "prototype_readiness_required": True,
            "prototype_support_bank_required": True,
            "silent_fallback_permitted": False,
            "output_status": BUILD_WEEK_PROTOTYPE_DEPLOYMENT_STATUS,
            "legacy_classifier_available": True,
            "b0_baseline_available": True,
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(
                    f"{field_name} must be {expected_value!r} for the Build Week "
                    "prototype mode"
                )

    @classmethod
    def read_json(cls, path: str | Path) -> BuildWeekPrototypeConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("Build Week prototype config must be a JSON object")
        values = dict(payload)
        if values.pop("schema_version", None) != BUILD_WEEK_PROTOTYPE_CONFIG_VERSION:
            raise ValueError("unsupported Build Week prototype config schema")
        allowed = {item.name for item in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                f"unknown Build Week prototype config fields: {sorted(unknown)}"
            )
        if isinstance(values.get("limitations"), list):
            values["limitations"] = tuple(values["limitations"])
        return cls(**values)  # type: ignore[arg-type]

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(self.to_manifest())

    def verify_artifacts(self) -> None:
        for path, expected in self.artifact_pins().values():
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(
                    f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
                )

    def artifact_pins(self) -> dict[str, tuple[Path, str]]:
        return {
            "reference_bank_readiness": (
                self.reference_bank_readiness,
                self.reference_bank_readiness_sha256,
            ),
            "support_manifest": (
                self.support_manifest,
                self.support_manifest_sha256,
            ),
            "reference_embeddings": (
                self.reference_embeddings,
                self.reference_embeddings_sha256,
            ),
            "candidate_score_evidence": (
                self.candidate_score_evidence,
                self.candidate_score_evidence_sha256,
            ),
            "prototype_policy": (
                self.prototype_policy,
                self.prototype_policy_sha256,
            ),
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": BUILD_WEEK_PROTOTYPE_CONFIG_VERSION,
            "classification_mode": self.classification_mode,
            "deployment_status": self.deployment_status,
            "workflow": self.workflow,
            "storage_backend": self.storage_backend,
            "s3_permitted": self.s3_permitted,
            "target": {
                "accepted_taxon_key": self.target_accepted_taxon_key,
                "scientific_name": self.target_scientific_name,
            },
            "invariants": {
                "target_always_scored": self.target_always_scored,
                "complete_regional_candidate_union_scored": (
                    self.complete_regional_candidate_union_scored
                ),
                "hierarchy_pruning_permitted": self.hierarchy_pruning_permitted,
                "spatial_crop_permitted": self.spatial_crop_permitted,
                "visual_input": self.visual_input,
                "prototype_readiness_required": self.prototype_readiness_required,
                "prototype_support_bank_required": (
                    self.prototype_support_bank_required
                ),
                "silent_fallback_permitted": self.silent_fallback_permitted,
                "output_status": self.output_status,
            },
            "frozen_identity": {
                "model_revision": self.model_revision,
                "preprocessing_version": self.preprocessing_version,
                "visual_input_version": self.visual_input_version,
                "classifier_fingerprint": self.classifier_fingerprint,
                "margin_policy_version": self.margin_policy_version,
            },
            "artifacts": {
                name: {"path": str(path), "sha256": sha256}
                for name, (path, sha256) in self.artifact_pins().items()
            },
            "diagnostic_baselines": {
                "legacy_classifier_available": self.legacy_classifier_available,
                "b0_baseline_available": self.b0_baseline_available,
            },
            "limitations": list(self.limitations),
        }


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
    ):
        raise ValueError(f"{field} must be sha256:<64 lowercase hex>")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be sha256:<64 lowercase hex>") from exc
    if value != value.lower():
        raise ValueError(f"{field} must use lowercase hex")
    return value


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value.strip()


__all__ = [
    "BUILD_WEEK_PROTOTYPE_CONFIG_VERSION",
    "BUILD_WEEK_PROTOTYPE_DEPLOYMENT_STATUS",
    "BUILD_WEEK_PROTOTYPE_VISUAL_INPUT",
    "BuildWeekPrototypeConfig",
]
