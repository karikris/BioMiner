"""Read-only revision impact analysis for dynamic pool artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint


DYNAMIC_REFERENCE_CHANGE_VERSION = "dynamic-reference-change-v1.0.0"
DYNAMIC_REFERENCE_REVISION_VERSION = "dynamic-reference-revision-v1.0.0"
DYNAMIC_POOL_DEPENDENCY_VERSION = "dynamic-pool-dependency-v1.0.0"
DYNAMIC_POOL_REVISION_IMPACT_VERSION = "dynamic-pool-revision-impact-v1.0.0"
DYNAMIC_POOL_IMPACT_PROJECTION_VERSION = "dynamic-pool-impact-projection-v1.0.0"

DYNAMIC_POOL_REVISION_IMPACT_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "revision_fingerprint": pl.String,
    "impact_fingerprint": pl.String,
    "pool_dependency_fingerprint": pl.String,
    "plan_id": pl.String,
    "plan_fingerprint": pl.String,
    "query_route": pl.String,
    "query_geo_cluster_id": pl.String,
    "candidate_taxon_keys": pl.List(pl.String),
    "member_reference_media_ids": pl.List(pl.String),
    "impact_status": pl.String,
    "direct_member_change": pl.Boolean,
    "affected_reference_media_ids": pl.List(pl.String),
    "changed_member_reference_media_ids": pl.List(pl.String),
    "changed_eligible_reference_media_ids": pl.List(pl.String),
    "impact_reasons": pl.List(pl.String),
    "expected_action": pl.String,
    "old_reference_bank_fingerprint": pl.String,
    "new_reference_bank_fingerprint": pl.String,
    "old_reference_geography_index_fingerprint": pl.String,
    "new_reference_geography_index_fingerprint": pl.String,
}


@dataclass(frozen=True, slots=True)
class DynamicReferenceChange:
    reference_media_id: str
    change_type: str
    old_taxon_key: str | None = None
    new_taxon_key: str | None = None
    old_route: str | None = None
    new_route: str | None = None
    old_geo_cluster_id: str | None = None
    new_geo_cluster_id: str | None = None
    old_global_anchor_eligible: bool = False
    new_global_anchor_eligible: bool = False
    old_local_anchor_eligible: bool = False
    new_local_anchor_eligible: bool = False
    schema_version: str = DYNAMIC_REFERENCE_CHANGE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_REFERENCE_CHANGE_VERSION:
            raise ValueError("unsupported dynamic reference change version")
        object.__setattr__(
            self,
            "reference_media_id",
            _required_text(self.reference_media_id, field="reference_media_id"),
        )
        change_type = _required_text(self.change_type, field="change_type")
        if change_type not in {"added", "removed", "modified"}:
            raise ValueError(f"unsupported reference change_type: {change_type}")
        object.__setattr__(self, "change_type", change_type)
        for field in (
            "old_taxon_key",
            "new_taxon_key",
            "old_route",
            "new_route",
            "old_geo_cluster_id",
            "new_geo_cluster_id",
        ):
            object.__setattr__(
                self,
                field,
                _optional_text(getattr(self, field), field=field),
            )
        for field in (
            "old_global_anchor_eligible",
            "new_global_anchor_eligible",
            "old_local_anchor_eligible",
            "new_local_anchor_eligible",
        ):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be a boolean")
        if change_type == "added" and (
            self.old_taxon_key is not None or self.old_route is not None
        ):
            raise ValueError("added reference cannot carry old selection context")
        if change_type == "removed" and (
            self.new_taxon_key is not None or self.new_route is not None
        ):
            raise ValueError("removed reference cannot carry new selection context")
        if change_type != "removed" and (
            self.new_taxon_key is None or self.new_route is None
        ):
            raise ValueError("current reference change requires new taxon and route")
        if change_type != "added" and (
            self.old_taxon_key is None or self.old_route is None
        ):
            raise ValueError("prior reference change requires old taxon and route")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class DynamicReferenceRevision:
    old_reference_bank_fingerprint: str
    new_reference_bank_fingerprint: str
    old_reference_geography_index_fingerprint: str
    new_reference_geography_index_fingerprint: str
    changes: tuple[DynamicReferenceChange, ...]
    schema_version: str = DYNAMIC_REFERENCE_REVISION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_REFERENCE_REVISION_VERSION:
            raise ValueError("unsupported dynamic reference revision version")
        for field in (
            "old_reference_bank_fingerprint",
            "new_reference_bank_fingerprint",
            "old_reference_geography_index_fingerprint",
            "new_reference_geography_index_fingerprint",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field=field))
        if self.old_reference_bank_fingerprint == self.new_reference_bank_fingerprint:
            raise ValueError("reference revision must change the bank fingerprint")
        changes = tuple(sorted(self.changes, key=lambda item: item.reference_media_id))
        if not changes:
            raise ValueError("reference revision requires at least one change")
        if any(not isinstance(item, DynamicReferenceChange) for item in changes):
            raise TypeError("revision changes must use DynamicReferenceChange")
        if len({item.reference_media_id for item in changes}) != len(changes):
            raise ValueError("reference revision repeats a reference media ID")
        object.__setattr__(self, "changes", changes)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": self.schema_version,
                "old_reference_bank_fingerprint": self.old_reference_bank_fingerprint,
                "new_reference_bank_fingerprint": self.new_reference_bank_fingerprint,
                "old_reference_geography_index_fingerprint": (
                    self.old_reference_geography_index_fingerprint
                ),
                "new_reference_geography_index_fingerprint": (
                    self.new_reference_geography_index_fingerprint
                ),
                "change_fingerprints": [item.fingerprint for item in self.changes],
            }
        )


@dataclass(frozen=True, slots=True)
class DynamicPoolDependency:
    plan_id: str
    plan_fingerprint: str
    query_route: str
    query_geo_cluster_id: str | None
    candidate_taxon_keys: tuple[str, ...]
    member_reference_media_ids: tuple[str, ...]
    member_fingerprints: tuple[str, ...]
    reference_bank_fingerprint: str
    reference_geography_index_fingerprint: str
    schema_version: str = DYNAMIC_POOL_DEPENDENCY_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_POOL_DEPENDENCY_VERSION:
            raise ValueError("unsupported dynamic pool dependency version")
        for field in ("plan_id", "query_route"):
            object.__setattr__(
                self, field, _required_text(getattr(self, field), field=field)
            )
        object.__setattr__(
            self,
            "query_geo_cluster_id",
            _optional_text(self.query_geo_cluster_id, field="query_geo_cluster_id"),
        )
        for field in (
            "plan_fingerprint",
            "reference_bank_fingerprint",
            "reference_geography_index_fingerprint",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field=field))
        candidates = _canonical_texts(
            self.candidate_taxon_keys, field="candidate_taxon_keys"
        )
        members = _canonical_texts(
            self.member_reference_media_ids,
            field="member_reference_media_ids",
        )
        fingerprints = tuple(
            sorted(
                _sha256(value, field="member_fingerprints")
                for value in self.member_fingerprints
            )
        )
        if not candidates or not members:
            raise ValueError("pool dependency requires candidates and members")
        if len(fingerprints) != len(members):
            raise ValueError("pool dependency member fingerprints must cover members")
        object.__setattr__(self, "candidate_taxon_keys", candidates)
        object.__setattr__(self, "member_reference_media_ids", members)
        object.__setattr__(self, "member_fingerprints", fingerprints)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class DynamicPoolImpactProjection:
    table: pl.DataFrame
    projection_fingerprint: str
    affected_plan_ids: tuple[str, ...]
    reusable_plan_ids: tuple[str, ...]
    changed_reference_ids_irrelevant_to_declared_pools: tuple[str, ...]


def identify_affected_reference_pools(
    revision: DynamicReferenceRevision,
    dependencies: Sequence[DynamicPoolDependency],
) -> DynamicPoolImpactProjection:
    """Identify member and newly eligible pool impacts without rebuilding."""

    if not isinstance(revision, DynamicReferenceRevision):
        raise TypeError("revision must be a DynamicReferenceRevision")
    pools = _normalized_pools(dependencies, revision=revision)
    relevant_change_ids: set[str] = set()
    rows = []
    for pool in pools:
        member_changes = sorted(
            set(pool.member_reference_media_ids)
            & {change.reference_media_id for change in revision.changes}
        )
        eligible_changes = sorted(
            change.reference_media_id
            for change in revision.changes
            if change.reference_media_id not in member_changes
            and _change_can_alter_pool_selection(change, pool)
        )
        relevant_change_ids.update(member_changes)
        relevant_change_ids.update(eligible_changes)
        affected_ids = sorted(set(member_changes) | set(eligible_changes))
        reasons = []
        if member_changes:
            reasons.append("declared_pool_member_changed")
        if eligible_changes:
            reasons.append("eligible_reference_can_change_pool_membership")
        affected = bool(affected_ids)
        base = {
            "revision_fingerprint": revision.fingerprint,
            "pool_dependency_fingerprint": pool.fingerprint,
            "plan_id": pool.plan_id,
            "plan_fingerprint": pool.plan_fingerprint,
            "query_route": pool.query_route,
            "query_geo_cluster_id": pool.query_geo_cluster_id,
            "candidate_taxon_keys": list(pool.candidate_taxon_keys),
            "member_reference_media_ids": list(pool.member_reference_media_ids),
            "impact_status": "affected" if affected else "reusable_as_is",
            "direct_member_change": bool(member_changes),
            "affected_reference_media_ids": affected_ids,
            "changed_member_reference_media_ids": member_changes,
            "changed_eligible_reference_media_ids": eligible_changes,
            "impact_reasons": reasons,
            "expected_action": (
                "rebuild_reference_pool" if affected else "reuse_pool_without_rebuild"
            ),
            "old_reference_bank_fingerprint": revision.old_reference_bank_fingerprint,
            "new_reference_bank_fingerprint": revision.new_reference_bank_fingerprint,
            "old_reference_geography_index_fingerprint": (
                revision.old_reference_geography_index_fingerprint
            ),
            "new_reference_geography_index_fingerprint": (
                revision.new_reference_geography_index_fingerprint
            ),
        }
        rows.append(
            {
                "schema_version": DYNAMIC_POOL_REVISION_IMPACT_VERSION,
                "impact_fingerprint": canonical_semantic_fingerprint(base),
                **base,
            }
        )
    table = pl.DataFrame(
        rows,
        schema=DYNAMIC_POOL_REVISION_IMPACT_SCHEMA,
        strict=True,
    ).sort("plan_id")
    validate_dynamic_pool_revision_impact(table)
    affected_plan_ids = tuple(
        table.filter(pl.col("impact_status") == "affected")["plan_id"]
    )
    reusable_plan_ids = tuple(
        table.filter(pl.col("impact_status") == "reusable_as_is")["plan_id"]
    )
    irrelevant = tuple(
        sorted(
            {change.reference_media_id for change in revision.changes}
            - relevant_change_ids
        )
    )
    projection_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_IMPACT_PROJECTION_VERSION,
            "revision_fingerprint": revision.fingerprint,
            "impact_fingerprints": table["impact_fingerprint"].to_list(),
            "affected_plan_ids": affected_plan_ids,
            "reusable_plan_ids": reusable_plan_ids,
            "irrelevant_changed_reference_ids": irrelevant,
        }
    )
    return DynamicPoolImpactProjection(
        table=table,
        projection_fingerprint=projection_fingerprint,
        affected_plan_ids=affected_plan_ids,
        reusable_plan_ids=reusable_plan_ids,
        changed_reference_ids_irrelevant_to_declared_pools=irrelevant,
    )


def validate_dynamic_pool_revision_impact(table: pl.DataFrame) -> None:
    if table.schema != DYNAMIC_POOL_REVISION_IMPACT_SCHEMA:
        raise ValueError("dynamic pool revision impact schema does not match contract")
    if table.is_empty():
        raise ValueError("dynamic pool revision impact must include declared pools")
    if table["plan_id"].n_unique() != table.height or not table.equals(
        table.sort("plan_id")
    ):
        raise ValueError("dynamic pool revision impact plan identities are invalid")
    for row in table.iter_rows(named=True):
        affected = row["impact_status"] == "affected"
        if row["impact_status"] not in {"affected", "reusable_as_is"}:
            raise ValueError("unsupported dynamic pool impact status")
        evidence = sorted(
            set(row["changed_member_reference_media_ids"])
            | set(row["changed_eligible_reference_media_ids"])
        )
        if (
            row["affected_reference_media_ids"] != evidence
            or bool(evidence) != affected
            or bool(row["changed_member_reference_media_ids"])
            != bool(row["direct_member_change"])
        ):
            raise ValueError("dynamic pool impact evidence is inconsistent")
        if not affected and (
            row["impact_reasons"]
            or row["expected_action"] != "reuse_pool_without_rebuild"
        ):
            raise ValueError("reusable pool carries rebuild evidence")
        base = {
            field: row[field]
            for field in DYNAMIC_POOL_REVISION_IMPACT_SCHEMA
            if field not in {"schema_version", "impact_fingerprint"}
        }
        if row["impact_fingerprint"] != canonical_semantic_fingerprint(base):
            raise ValueError("dynamic pool revision impact fingerprint mismatch")


def _change_can_alter_pool_selection(
    change: DynamicReferenceChange,
    pool: DynamicPoolDependency,
) -> bool:
    contexts = (
        (
            change.new_taxon_key,
            change.new_route,
            change.new_geo_cluster_id,
            change.new_global_anchor_eligible,
            change.new_local_anchor_eligible,
        ),
    )
    for taxon, route, geo, global_eligible, local_eligible in contexts:
        if taxon not in pool.candidate_taxon_keys or route != pool.query_route:
            continue
        if global_eligible:
            return True
        if local_eligible and geo is not None and geo == pool.query_geo_cluster_id:
            return True
    return False


def _normalized_pools(
    dependencies: Sequence[DynamicPoolDependency],
    *,
    revision: DynamicReferenceRevision,
) -> tuple[DynamicPoolDependency, ...]:
    pools = tuple(dependencies)
    if not pools or any(not isinstance(pool, DynamicPoolDependency) for pool in pools):
        raise ValueError("dynamic pool impact requires typed dependencies")
    ordered = tuple(sorted(pools, key=lambda pool: pool.plan_id))
    if len({pool.plan_id for pool in ordered}) != len(ordered):
        raise ValueError("dynamic pool dependencies repeat a plan ID")
    for pool in ordered:
        if pool.reference_bank_fingerprint != revision.old_reference_bank_fingerprint:
            raise ValueError("pool dependency has stale reference-bank binding")
        if (
            pool.reference_geography_index_fingerprint
            != revision.old_reference_geography_index_fingerprint
        ):
            raise ValueError("pool dependency has stale geography-index binding")
    return ordered


def _canonical_texts(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    normalized = tuple(sorted({_required_text(value, field=field) for value in values}))
    return normalized


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field).casefold()
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field} must be a sha256 fingerprint")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a sha256 fingerprint") from exc
    return text


__all__ = [
    "DYNAMIC_POOL_DEPENDENCY_VERSION",
    "DYNAMIC_POOL_IMPACT_PROJECTION_VERSION",
    "DYNAMIC_POOL_REVISION_IMPACT_SCHEMA",
    "DYNAMIC_POOL_REVISION_IMPACT_VERSION",
    "DYNAMIC_REFERENCE_CHANGE_VERSION",
    "DYNAMIC_REFERENCE_REVISION_VERSION",
    "DynamicPoolDependency",
    "DynamicPoolImpactProjection",
    "DynamicReferenceChange",
    "DynamicReferenceRevision",
    "identify_affected_reference_pools",
    "validate_dynamic_pool_revision_impact",
]
