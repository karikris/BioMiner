from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


VISION_BUCKET_POLICY_SCHEMA_VERSION = "vision-bucket-policy-v1"
DEFAULT_VISION_BUCKET_POLICY_PATH = Path(__file__).resolve().parents[3] / "config" / "vision_bucket_policy.json"


@dataclass(frozen=True)
class VisionBucketPolicy:
    schema_version: str = VISION_BUCKET_POLICY_SCHEMA_VERSION
    high_confidence_species_top1_score: float = 0.70
    minimum_species_margin: float = 0.05
    minimum_family_margin: float = 0.05
    low_confidence_species_score: float = 0.35
    high_detector_score: float = 0.80
    missing_bioclip_review_priority: int = 100
    hard_negative_review_priority: int = 95
    metadata_conflict_review_priority: int = 90
    family_species_conflict_review_priority: int = 85
    multi_object_conflict_review_priority: int = 80
    high_detector_weak_species_review_priority: int = 75
    low_species_margin_review_priority: int = 70
    low_family_margin_review_priority: int = 65
    geospatial_prior_conflict_review_priority: int = 60
    clean_confident_review_priority: int = 10


def load_vision_bucket_policy(path: str | Path | None = None) -> VisionBucketPolicy:
    payload = _read_policy_payload(DEFAULT_VISION_BUCKET_POLICY_PATH if path is None else Path(path))
    policy = VisionBucketPolicy(**payload)
    _validate_policy(policy)
    return policy


def _read_policy_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if path == DEFAULT_VISION_BUCKET_POLICY_PATH:
            return {}
        raise
    if not isinstance(payload, dict):
        raise ValueError("vision bucket policy must be a JSON object")
    return payload


def _validate_policy(policy: VisionBucketPolicy) -> None:
    if policy.schema_version != VISION_BUCKET_POLICY_SCHEMA_VERSION:
        raise ValueError(
            "vision bucket policy schema_version must be "
            f"{VISION_BUCKET_POLICY_SCHEMA_VERSION!r}, got {policy.schema_version!r}"
        )
    for field_name in (
        "high_confidence_species_top1_score",
        "minimum_species_margin",
        "minimum_family_margin",
        "low_confidence_species_score",
        "high_detector_score",
    ):
        _validate_unit_interval(getattr(policy, field_name), field_name)
    if policy.low_confidence_species_score > policy.high_confidence_species_top1_score:
        raise ValueError("low_confidence_species_score must be <= high_confidence_species_top1_score")
    for field_name in (
        "missing_bioclip_review_priority",
        "hard_negative_review_priority",
        "metadata_conflict_review_priority",
        "family_species_conflict_review_priority",
        "multi_object_conflict_review_priority",
        "high_detector_weak_species_review_priority",
        "low_species_margin_review_priority",
        "low_family_margin_review_priority",
        "geospatial_prior_conflict_review_priority",
        "clean_confident_review_priority",
    ):
        _validate_priority(getattr(policy, field_name), field_name)


def _validate_unit_interval(value: float, field_name: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")


def _validate_priority(value: int, field_name: str) -> None:
    if not 0 <= int(value) <= 100:
        raise ValueError(f"{field_name} must be between 0 and 100")


__all__ = [
    "DEFAULT_VISION_BUCKET_POLICY_PATH",
    "VISION_BUCKET_POLICY_SCHEMA_VERSION",
    "VisionBucketPolicy",
    "load_vision_bucket_policy",
]
