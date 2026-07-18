"""Typed, fingerprinted settings for the dynamic-pooling workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path

from biominer.bioclip.dynamic_pool_fusion import RAW_FUSION_METHODS
from biominer.bioclip.dynamic_pool_policy import (
    DynamicReferencePoolPolicy,
    default_dynamic_reference_pool_policy,
)
from biominer.candidates.strategy_ablation import CANDIDATE_STRATEGIES
from biominer.common.semantic_hash import canonical_semantic_fingerprint


DYNAMIC_POOLING_SETTINGS_VERSION = "dynamic-pooling-settings-v1.0.0"
DYNAMIC_POOLING_SETTINGS_FILE = "dynamic_pooling_settings.json"

_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_strategy",
        "candidate_strategy_selection_fingerprint",
        "fusion_method",
        "fusion_selection_fingerprint",
        "reference_pool_policy",
        "review_budget",
        "review_random_seed",
        "flickr_embedding_batch_size",
        "vector_score_batch_size",
        "maximum_matrix_cache_bytes",
        "release_requires_human_review",
        "representative_probability_sampling_required",
        "selective_rerun_enabled",
        "missing_geography_is_biological_absence",
        "raw_scores_are_probabilities",
    }
)


@dataclass(frozen=True, slots=True)
class DynamicPoolingSettings:
    """One fail-closed configuration shared by dynamic workflow commands."""

    schema_version: str = DYNAMIC_POOLING_SETTINGS_VERSION
    candidate_strategy: str | None = None
    candidate_strategy_selection_fingerprint: str | None = None
    fusion_method: str | None = None
    fusion_selection_fingerprint: str | None = None
    reference_pool_policy: DynamicReferencePoolPolicy = field(
        default_factory=default_dynamic_reference_pool_policy
    )
    review_budget: int = 50
    review_random_seed: int = 42
    flickr_embedding_batch_size: int = 64
    vector_score_batch_size: int = 512
    maximum_matrix_cache_bytes: int = 512 * 1024 * 1024
    release_requires_human_review: bool = True
    representative_probability_sampling_required: bool = True
    selective_rerun_enabled: bool = True
    missing_geography_is_biological_absence: bool = False
    raw_scores_are_probabilities: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_POOLING_SETTINGS_VERSION:
            raise ValueError("unsupported dynamic-pooling settings version")
        if not isinstance(self.reference_pool_policy, DynamicReferencePoolPolicy):
            raise TypeError("reference_pool_policy must be typed")
        candidate_strategy = _optional_text(self.candidate_strategy)
        candidate_evidence = _optional_text(
            self.candidate_strategy_selection_fingerprint
        )
        fusion_method = _optional_text(self.fusion_method)
        fusion_evidence = _optional_text(self.fusion_selection_fingerprint)
        object.__setattr__(self, "candidate_strategy", candidate_strategy)
        object.__setattr__(
            self,
            "candidate_strategy_selection_fingerprint",
            candidate_evidence,
        )
        object.__setattr__(self, "fusion_method", fusion_method)
        object.__setattr__(self, "fusion_selection_fingerprint", fusion_evidence)
        _selection(
            candidate_strategy,
            candidate_evidence,
            allowed=CANDIDATE_STRATEGIES,
            field="candidate_strategy",
        )
        _selection(
            fusion_method,
            fusion_evidence,
            allowed=frozenset(RAW_FUSION_METHODS),
            field="fusion_method",
        )
        for name in (
            "review_budget",
            "flickr_embedding_batch_size",
            "vector_score_batch_size",
            "maximum_matrix_cache_bytes",
        ):
            _positive_int(getattr(self, name), field=name)
        _uint64(self.review_random_seed, field="review_random_seed")
        required_true = (
            "release_requires_human_review",
            "representative_probability_sampling_required",
            "selective_rerun_enabled",
        )
        for name in required_true:
            if getattr(self, name) is not True:
                raise ValueError(f"dynamic-pooling safety requires {name}=true")
        required_false = (
            "missing_geography_is_biological_absence",
            "raw_scores_are_probabilities",
        )
        for name in required_false:
            if getattr(self, name) is not False:
                raise ValueError(f"dynamic-pooling safety requires {name}=false")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(self.identity_payload())

    @property
    def selection_status(self) -> str:
        selected_count = sum(
            value is not None for value in (self.candidate_strategy, self.fusion_method)
        )
        if selected_count == 0:
            return "unselected"
        if selected_count == 1:
            return "partially_selected_with_bound_evidence"
        return "selected_with_bound_evidence"

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate_strategy": self.candidate_strategy,
            "candidate_strategy_selection_fingerprint": (
                self.candidate_strategy_selection_fingerprint
            ),
            "fusion_method": self.fusion_method,
            "fusion_selection_fingerprint": self.fusion_selection_fingerprint,
            "reference_pool_policy": self.reference_pool_policy.to_dict(),
            "review_budget": self.review_budget,
            "review_random_seed": self.review_random_seed,
            "flickr_embedding_batch_size": self.flickr_embedding_batch_size,
            "vector_score_batch_size": self.vector_score_batch_size,
            "maximum_matrix_cache_bytes": self.maximum_matrix_cache_bytes,
            "release_requires_human_review": self.release_requires_human_review,
            "representative_probability_sampling_required": (
                self.representative_probability_sampling_required
            ),
            "selective_rerun_enabled": self.selective_rerun_enabled,
            "missing_geography_is_biological_absence": (
                self.missing_geography_is_biological_absence
            ),
            "raw_scores_are_probabilities": self.raw_scores_are_probabilities,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "settings_fingerprint": self.fingerprint,
            "selection_status": self.selection_status,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> DynamicPoolingSettings:
        if not isinstance(value, Mapping):
            raise TypeError("dynamic-pooling settings must be a mapping")
        expected = _IDENTITY_FIELDS | {"settings_fingerprint", "selection_status"}
        if set(value) != expected:
            raise ValueError("dynamic-pooling settings fields do not match")
        pool_policy = value["reference_pool_policy"]
        if not isinstance(pool_policy, Mapping):
            raise ValueError("reference_pool_policy must be a mapping")
        settings = cls(
            schema_version=_text(value["schema_version"], field="schema_version"),
            candidate_strategy=_optional_text(value["candidate_strategy"]),
            candidate_strategy_selection_fingerprint=_optional_text(
                value["candidate_strategy_selection_fingerprint"]
            ),
            fusion_method=_optional_text(value["fusion_method"]),
            fusion_selection_fingerprint=_optional_text(
                value["fusion_selection_fingerprint"]
            ),
            reference_pool_policy=DynamicReferencePoolPolicy.from_mapping(pool_policy),
            review_budget=_mapping_int(value, "review_budget"),
            review_random_seed=_mapping_int(value, "review_random_seed"),
            flickr_embedding_batch_size=_mapping_int(
                value, "flickr_embedding_batch_size"
            ),
            vector_score_batch_size=_mapping_int(value, "vector_score_batch_size"),
            maximum_matrix_cache_bytes=_mapping_int(
                value, "maximum_matrix_cache_bytes"
            ),
            release_requires_human_review=_mapping_bool(
                value, "release_requires_human_review"
            ),
            representative_probability_sampling_required=_mapping_bool(
                value, "representative_probability_sampling_required"
            ),
            selective_rerun_enabled=_mapping_bool(value, "selective_rerun_enabled"),
            missing_geography_is_biological_absence=_mapping_bool(
                value, "missing_geography_is_biological_absence"
            ),
            raw_scores_are_probabilities=_mapping_bool(
                value, "raw_scores_are_probabilities"
            ),
        )
        if value["settings_fingerprint"] != settings.fingerprint:
            raise ValueError("dynamic-pooling settings fingerprint mismatch")
        if value["selection_status"] != settings.selection_status:
            raise ValueError("dynamic-pooling settings selection status mismatch")
        return settings


def load_dynamic_pooling_settings(path: str | Path) -> DynamicPoolingSettings:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("dynamic-pooling settings JSON is invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError("dynamic-pooling settings JSON must contain an object")
    return DynamicPoolingSettings.from_mapping(value)


def write_dynamic_pooling_settings(
    settings: DynamicPoolingSettings,
    output: str | Path,
) -> Path:
    if not isinstance(settings, DynamicPoolingSettings):
        raise TypeError("settings must be DynamicPoolingSettings")
    destination = Path(output)
    if destination.suffix.casefold() != ".json":
        destination /= DYNAMIC_POOLING_SETTINGS_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(settings.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _selection(
    value: str | None,
    evidence_fingerprint: str | None,
    *,
    allowed: frozenset[str],
    field: str,
) -> None:
    selected = _optional_text(value)
    evidence = _optional_text(evidence_fingerprint)
    if selected is None:
        if evidence is not None:
            raise ValueError(f"{field} evidence requires a selected value")
        return
    if selected not in allowed:
        raise ValueError(f"unsupported {field}: {selected!r}")
    if evidence is None:
        raise ValueError(f"selected {field} requires evidence fingerprint")
    _sha256(evidence, field=f"{field}_selection_fingerprint")


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _text(value, field="optional setting").casefold()


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonblank text")
    return value.strip()


def _mapping_int(value: Mapping[str, object], field: str) -> int:
    item = value[field]
    if isinstance(item, bool) or not isinstance(item, int):
        raise TypeError(f"{field} must be an integer")
    return item


def _mapping_bool(value: Mapping[str, object], field: str) -> bool:
    item = value[field]
    if not isinstance(item, bool):
        raise TypeError(f"{field} must be Boolean")
    return item


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _uint64(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
        raise ValueError(f"{field} must be an unsigned 64-bit integer")
    return value


def _sha256(value: object, *, field: str) -> str:
    text = _text(value, field=field).casefold()
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field} must be a sha256 fingerprint")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a sha256 fingerprint") from exc
    return text


__all__ = [
    "DYNAMIC_POOLING_SETTINGS_FILE",
    "DYNAMIC_POOLING_SETTINGS_VERSION",
    "DynamicPoolingSettings",
    "load_dynamic_pooling_settings",
    "write_dynamic_pooling_settings",
]
