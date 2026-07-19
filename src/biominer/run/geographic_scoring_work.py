"""Deterministic cache-local ordering for geographic BioCLIP scoring work."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re

from biominer.bioclip.dynamic_pool_contracts import DYNAMIC_POOL_GEOGRAPHIC_SCOPES
from biominer.bioclip.matrix_cache import (
    candidate_pool_signature as build_candidate_pool_signature,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.readiness import REFERENCE_ROUTES
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
)


GEOGRAPHIC_SCORING_WORK_SCHEMA_VERSION = "geographic-scoring-work-v1"
GEOGRAPHIC_SCORING_ORDER_VERSION = "geographic-scoring-order-v1"

_VISUAL_INPUT_KINDS = frozenset(
    {
        RAW_FULL_IMAGE_KIND,
        FOCUSED_FULL_FRAME_KIND,
        MASKED_FULL_FRAME_KIND,
        MULTI_OBJECT_FULL_FRAME_KIND,
    }
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class GeographicScoringWorkItem:
    """One scoring descriptor whose biological inputs are already immutable."""

    work_id: str
    route: str
    visual_input_kind: str
    family_partition: str
    geographic_scope: str
    candidate_matrix_signature: str
    pool_matrix_signature: str
    payload_fingerprint: str
    result_order_key: tuple[str, ...]
    candidate_pool_signature: str | None = None
    work_fingerprint: str | None = None

    def __post_init__(self) -> None:
        work_id = _required_text(self.work_id, field="work_id")
        route = _route(self.route)
        input_kind = _visual_input_kind(self.visual_input_kind)
        partition = _required_text(self.family_partition, field="family_partition")
        scope = _geographic_scope(self.geographic_scope)
        candidate_signature = _sha256(
            self.candidate_matrix_signature,
            field="candidate_matrix_signature",
        )
        pool_signature = _sha256(
            self.pool_matrix_signature,
            field="pool_matrix_signature",
        )
        payload_fingerprint = _sha256(
            self.payload_fingerprint,
            field="payload_fingerprint",
        )
        result_key = _result_order_key(self.result_order_key)
        combined_signature = build_candidate_pool_signature(
            candidate_signature,
            (pool_signature,),
        )
        if (
            self.candidate_pool_signature is not None
            and _sha256(
                self.candidate_pool_signature,
                field="candidate_pool_signature",
            )
            != combined_signature
        ):
            raise ValueError(
                "candidate_pool_signature does not match candidate and pool matrices"
            )
        base = {
            "schema_version": GEOGRAPHIC_SCORING_WORK_SCHEMA_VERSION,
            "work_id": work_id,
            "route": route,
            "visual_input_kind": input_kind,
            "family_partition": partition,
            "geographic_scope": scope,
            "candidate_matrix_signature": candidate_signature,
            "pool_matrix_signature": pool_signature,
            "candidate_pool_signature": combined_signature,
            "payload_fingerprint": payload_fingerprint,
            "result_order_key": list(result_key),
        }
        fingerprint = canonical_semantic_fingerprint(base)
        if (
            self.work_fingerprint is not None
            and _sha256(
                self.work_fingerprint,
                field="work_fingerprint",
            )
            != fingerprint
        ):
            raise ValueError("work_fingerprint does not match scoring work")
        object.__setattr__(self, "work_id", work_id)
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "visual_input_kind", input_kind)
        object.__setattr__(self, "family_partition", partition)
        object.__setattr__(self, "geographic_scope", scope)
        object.__setattr__(self, "candidate_matrix_signature", candidate_signature)
        object.__setattr__(self, "pool_matrix_signature", pool_signature)
        object.__setattr__(self, "payload_fingerprint", payload_fingerprint)
        object.__setattr__(self, "result_order_key", result_key)
        object.__setattr__(self, "candidate_pool_signature", combined_signature)
        object.__setattr__(self, "work_fingerprint", fingerprint)

    @property
    def cache_locality_key(self) -> tuple[str, str, str, str, str]:
        """The exact primary key required by the scoring work contract."""

        assert self.candidate_pool_signature is not None
        return (
            self.route,
            self.visual_input_kind,
            self.family_partition,
            self.geographic_scope,
            self.candidate_pool_signature,
        )

    @property
    def execution_sort_key(self) -> tuple[object, ...]:
        """Primary locality fields followed only by deterministic tie breakers."""

        return (
            *self.cache_locality_key,
            self.candidate_matrix_signature,
            self.pool_matrix_signature,
            self.result_order_key,
            self.work_id,
        )

    @property
    def canonical_result_sort_key(self) -> tuple[object, ...]:
        """Result publication order, independent of execution scheduling."""

        return (*self.result_order_key, self.work_id)


@dataclass(frozen=True, slots=True)
class GeographicScoringOrderMetrics:
    """Measured cache-locality properties of one deterministic work order."""

    work_item_count: int
    unique_route_count: int
    unique_visual_input_kind_count: int
    unique_family_partition_count: int
    unique_geographic_scope_count: int
    unique_candidate_matrix_signature_count: int
    unique_pool_matrix_signature_count: int
    unique_candidate_pool_signature_count: int
    cache_locality_run_count: int
    candidate_matrix_run_count: int
    pool_matrix_run_count: int
    candidate_pool_signature_run_count: int
    candidate_matrix_reuse_opportunity_count: int
    pool_matrix_reuse_opportunity_count: int
    adjacent_candidate_matrix_reuse_count: int
    adjacent_pool_matrix_reuse_count: int
    execution_order_changed: bool

    def as_dict(self) -> dict[str, bool | int]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class GeographicScoringWorkOrder:
    """Cache-local execution order plus a stable publication-order projection."""

    items: tuple[GeographicScoringWorkItem, ...]
    metrics: GeographicScoringOrderMetrics
    ordering_fingerprint: str

    def canonical_result_order(self) -> tuple[GeographicScoringWorkItem, ...]:
        return tuple(
            sorted(self.items, key=lambda item: item.canonical_result_sort_key)
        )


def sort_geographic_scoring_work(
    items: Sequence[GeographicScoringWorkItem],
) -> GeographicScoringWorkOrder:
    """Sort work for matrix reuse without changing its scientific identities."""

    if isinstance(items, str | bytes) or not isinstance(items, Sequence):
        raise TypeError("geographic scoring work must be a sequence")
    source = tuple(items)
    if any(not isinstance(item, GeographicScoringWorkItem) for item in source):
        raise TypeError(
            "geographic scoring work must contain GeographicScoringWorkItem values"
        )
    work_ids = [item.work_id for item in source]
    if len(work_ids) != len(set(work_ids)):
        raise ValueError("geographic scoring work IDs must be unique")
    work_fingerprints = [item.work_fingerprint for item in source]
    if len(work_fingerprints) != len(set(work_fingerprints)):
        raise ValueError("geographic scoring work fingerprints must be unique")

    ordered = tuple(sorted(source, key=lambda item: item.execution_sort_key))
    ordering_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": GEOGRAPHIC_SCORING_ORDER_VERSION,
            "work_fingerprints": [item.work_fingerprint for item in ordered],
        }
    )
    return GeographicScoringWorkOrder(
        items=ordered,
        metrics=_ordering_metrics(source, ordered),
        ordering_fingerprint=ordering_fingerprint,
    )


def _ordering_metrics(
    source: tuple[GeographicScoringWorkItem, ...],
    ordered: tuple[GeographicScoringWorkItem, ...],
) -> GeographicScoringOrderMetrics:
    candidate_signatures = {item.candidate_matrix_signature for item in ordered}
    pool_signatures = {item.pool_matrix_signature for item in ordered}
    candidate_pool_signatures = {item.candidate_pool_signature for item in ordered}
    return GeographicScoringOrderMetrics(
        work_item_count=len(ordered),
        unique_route_count=len({item.route for item in ordered}),
        unique_visual_input_kind_count=len(
            {item.visual_input_kind for item in ordered}
        ),
        unique_family_partition_count=len({item.family_partition for item in ordered}),
        unique_geographic_scope_count=len({item.geographic_scope for item in ordered}),
        unique_candidate_matrix_signature_count=len(candidate_signatures),
        unique_pool_matrix_signature_count=len(pool_signatures),
        unique_candidate_pool_signature_count=len(candidate_pool_signatures),
        cache_locality_run_count=_run_count(
            ordered,
            key=lambda item: item.cache_locality_key,
        ),
        candidate_matrix_run_count=_run_count(
            ordered,
            key=lambda item: item.candidate_matrix_signature,
        ),
        pool_matrix_run_count=_run_count(
            ordered,
            key=lambda item: item.pool_matrix_signature,
        ),
        candidate_pool_signature_run_count=_run_count(
            ordered,
            key=lambda item: item.candidate_pool_signature,
        ),
        candidate_matrix_reuse_opportunity_count=(
            len(ordered) - len(candidate_signatures)
        ),
        pool_matrix_reuse_opportunity_count=len(ordered) - len(pool_signatures),
        adjacent_candidate_matrix_reuse_count=_adjacent_equal_count(
            ordered,
            key=lambda item: item.candidate_matrix_signature,
        ),
        adjacent_pool_matrix_reuse_count=_adjacent_equal_count(
            ordered,
            key=lambda item: item.pool_matrix_signature,
        ),
        execution_order_changed=tuple(item.work_id for item in source)
        != tuple(item.work_id for item in ordered),
    )


def _run_count(
    items: Sequence[GeographicScoringWorkItem],
    *,
    key: Callable[[GeographicScoringWorkItem], object],
) -> int:
    if not items:
        return 0
    return 1 + sum(key(left) != key(right) for left, right in zip(items, items[1:]))


def _adjacent_equal_count(
    items: Sequence[GeographicScoringWorkItem],
    *,
    key: Callable[[GeographicScoringWorkItem], object],
) -> int:
    return sum(key(left) == key(right) for left, right in zip(items, items[1:]))


def _result_order_key(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError("result_order_key must be a sequence")
    result = tuple(_required_text(value, field="result_order_key") for value in values)
    if not result:
        raise ValueError("result_order_key must not be empty")
    return result


def _route(value: object) -> str:
    route = _required_text(value, field="route")
    if route not in REFERENCE_ROUTES:
        raise ValueError(f"unsupported scoring work route: {route}")
    return route


def _visual_input_kind(value: object) -> str:
    kind = _required_text(value, field="visual_input_kind")
    if kind not in _VISUAL_INPUT_KINDS:
        raise ValueError(f"unsupported scoring work visual_input_kind: {kind}")
    return kind


def _geographic_scope(value: object) -> str:
    scope = _required_text(value, field="geographic_scope")
    if scope not in DYNAMIC_POOL_GEOGRAPHIC_SCOPES:
        raise ValueError(f"unsupported scoring work geographic_scope: {scope}")
    return scope


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a canonical sha256 fingerprint")
    return text


__all__ = [
    "GEOGRAPHIC_SCORING_ORDER_VERSION",
    "GEOGRAPHIC_SCORING_WORK_SCHEMA_VERSION",
    "GeographicScoringOrderMetrics",
    "GeographicScoringWorkItem",
    "GeographicScoringWorkOrder",
    "sort_geographic_scoring_work",
]
