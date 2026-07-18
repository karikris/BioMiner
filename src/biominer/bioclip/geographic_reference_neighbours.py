"""Precision-aware lookup memberships for local and global reference fallback."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re

import polars as pl

from biominer.bioclip.global_reference_anchors import (
    global_reference_anchors_artifact_fingerprint,
    validate_global_reference_anchors,
)
from biominer.bioclip.reference_geography_index import (
    reference_geography_index_artifact_fingerprint,
    validate_reference_geography_index,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.geography import CellGrid, default_cell_grid
from biominer.references.normalized_geography import (
    normalized_reference_geography_artifact_fingerprint,
    validate_normalized_reference_geography,
)
from biominer.storage.parquet import write_parquet


GEOGRAPHIC_REFERENCE_NEIGHBOURS_SCHEMA_VERSION = (
    "geographic-reference-neighbours-v1.0.0"
)
GEOGRAPHIC_REFERENCE_NEIGHBOURS_FILE = "geographic_reference_neighbours.parquet"
GEOGRAPHIC_REFERENCE_NEIGHBOUR_POLICY_VERSION = (
    "geographic-reference-neighbour-policy-v1.0.0"
)

REFERENCE_NEIGHBOUR_SCOPES = frozenset(
    {
        "exact_supported_cell",
        "neighbouring_supported_cell",
        "parent_regional_cell",
        "parent_coarse_cell",
        "bioregion",
        "country",
        "continent",
        "global",
    }
)
_SCOPE_FALLBACK_LEVEL = {
    "exact_supported_cell": 0,
    "neighbouring_supported_cell": 1,
    "parent_regional_cell": 2,
    "parent_coarse_cell": 3,
    "bioregion": 4,
    "country": 5,
    "continent": 6,
    "global": 7,
}
_SCOPE_REASON = {
    "exact_supported_cell": "reference_supports_lookup_cell_exactly",
    "neighbouring_supported_cell": "lookup_cell_neighbours_reference_supported_cell",
    "parent_regional_cell": "reference_local_cell_descends_from_lookup_regional_cell",
    "parent_coarse_cell": "reference_supported_cell_descends_from_lookup_coarse_cell",
    "bioregion": "reference_has_source_bioregion",
    "country": "reference_has_source_country",
    "continent": "reference_has_source_continent",
    "global": "reference_selected_as_global_anchor",
}
_CELL_SCOPES = frozenset(
    {
        "exact_supported_cell",
        "neighbouring_supported_cell",
        "parent_regional_cell",
        "parent_coarse_cell",
    }
)
_NAMED_SCOPES = frozenset({"bioregion", "country", "continent", "global"})
_GEO_QUALITIES = frozenset({"local", "regional", "coarse"})
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MEMBERSHIP_ID_PATTERN = re.compile(r"geographic-reference-neighbour:[0-9a-f]{64}\Z")
_GRAIN = (
    "reference_geography_row_fingerprint",
    "lookup_scope",
    "lookup_key",
)
_SORT = (
    "fallback_level",
    "lookup_scope",
    "lookup_key",
    "accepted_taxon_key",
    "route",
    "lookup_rank",
    "reference_observation_id",
    "reference_media_id",
    "embedding_fingerprint",
)


@dataclass(frozen=True, slots=True)
class GeographicReferenceNeighbourPolicy:
    """Fixed, bounded fallback expansion over the accepted cell hierarchy."""

    neighbour_grid_distance: int = 1
    version: str = GEOGRAPHIC_REFERENCE_NEIGHBOUR_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.neighbour_grid_distance != 1 or isinstance(
            self.neighbour_grid_distance, bool
        ):
            raise ValueError("neighbour_grid_distance must be exactly one")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("geographic neighbour policy version must be nonblank")
        object.__setattr__(self, "version", self.version.strip())

    @property
    def fingerprint(self) -> str:
        """Return the immutable fallback-policy identity."""

        return canonical_semantic_fingerprint(
            {
                "schema_version": GEOGRAPHIC_REFERENCE_NEIGHBOURS_SCHEMA_VERSION,
                "version": self.version,
                "neighbour_grid_distance": self.neighbour_grid_distance,
                "fallback_levels": _SCOPE_FALLBACK_LEVEL,
                "cell_expansion": "finest_supported_cell_only",
                "embedding_materialization": False,
                "geography_is_identity_authority": False,
            }
        )


def geographic_reference_neighbours_schema() -> dict[str, pl.DataType]:
    """Return the closed lookup-membership schema."""

    return {
        "schema_version": pl.String,
        "registry_version": pl.String,
        "reference_bank_version": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "route": pl.String,
        "visual_domain": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "duplicate_group_id": pl.String,
        "source": pl.String,
        "embedding_fingerprint": pl.String,
        "reference_geography_row_fingerprint": pl.String,
        "normalized_geography_row_fingerprint": pl.String,
        "coordinate_quality": pl.String,
        "geography_unavailable_reason": pl.String,
        "cell_grid_name": pl.String,
        "cell_grid_version": pl.String,
        "coarse_cell_resolution": pl.UInt8,
        "regional_cell_resolution": pl.UInt8,
        "local_cell_resolution": pl.UInt8,
        "supported_cell_level": pl.String,
        "supported_cell_resolution": pl.UInt8,
        "supported_cell_id": pl.String,
        "coarse_cell_id": pl.String,
        "regional_cell_id": pl.String,
        "local_cell_id": pl.String,
        "country_code": pl.String,
        "bioregion": pl.String,
        "continent_code": pl.String,
        "is_global_anchor": pl.Boolean,
        "lookup_scope": pl.String,
        "lookup_key": pl.String,
        "lookup_cell_level": pl.String,
        "lookup_cell_resolution": pl.UInt8,
        "lookup_cell_id": pl.String,
        "neighbour_grid_distance": pl.UInt8,
        "fallback_level": pl.UInt8,
        "inclusion_reason": pl.String,
        "membership_id": pl.String,
        "lookup_rank": pl.UInt32,
        "membership_policy_fingerprint": pl.String,
        "reference_geography_index_fingerprint": pl.String,
        "normalized_reference_geography_fingerprint": pl.String,
        "global_reference_anchors_fingerprint": pl.String,
        "row_fingerprint": pl.String,
    }


def build_geographic_reference_neighbours(
    reference_geography_index: pl.DataFrame,
    normalized_reference_geography: pl.DataFrame,
    global_reference_anchors: pl.DataFrame,
    *,
    policy: GeographicReferenceNeighbourPolicy | None = None,
    grid: CellGrid | None = None,
) -> pl.DataFrame:
    """Materialize exact, adjacent, parent, named and global lookup memberships."""

    validate_reference_geography_index(reference_geography_index)
    validate_normalized_reference_geography(normalized_reference_geography)
    validate_global_reference_anchors(global_reference_anchors)
    active_policy = policy or GeographicReferenceNeighbourPolicy()
    if not isinstance(active_policy, GeographicReferenceNeighbourPolicy):
        raise TypeError("policy must be GeographicReferenceNeighbourPolicy")

    index_fingerprint = reference_geography_index_artifact_fingerprint(
        reference_geography_index
    )
    normalized_fingerprint = normalized_reference_geography_artifact_fingerprint(
        normalized_reference_geography
    )
    anchors_fingerprint = global_reference_anchors_artifact_fingerprint(
        global_reference_anchors
    )
    _validate_anchor_parent(
        global_reference_anchors,
        index_fingerprint=index_fingerprint,
        known_index_rows=set(reference_geography_index["row_fingerprint"].to_list()),
    )
    geography_by_observation = {
        str(row["reference_observation_id"]): row
        for row in normalized_reference_geography.iter_rows(named=True)
    }
    _require_geography_coverage(
        reference_geography_index,
        geography_by_observation=geography_by_observation,
    )
    anchor_rows = set(
        global_reference_anchors["reference_geography_row_fingerprint"].to_list()
    )
    cell_rows = reference_geography_index.filter(pl.col("local_anchor_eligible"))
    backend = _validated_grid(
        grid,
        cell_rows=cell_rows,
        geography_by_observation=geography_by_observation,
    )

    rows: list[dict[str, object]] = []
    for index_row in reference_geography_index.iter_rows(named=True):
        geography = geography_by_observation[str(index_row["reference_observation_id"])]
        _validate_index_geography(index_row, geography=geography)
        is_global_anchor = index_row["row_fingerprint"] in anchor_rows
        rows.extend(
            _memberships_for_reference(
                index_row,
                geography=geography,
                is_global_anchor=is_global_anchor,
                policy=active_policy,
                grid=backend,
                index_fingerprint=index_fingerprint,
                normalized_fingerprint=normalized_fingerprint,
                anchors_fingerprint=anchors_fingerprint,
            )
        )

    rows.sort(key=_pre_rank_sort_key)
    ranks: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for row in rows:
        bucket = (
            str(row["lookup_scope"]),
            str(row["lookup_key"]),
            str(row["accepted_taxon_key"]),
            str(row["route"]),
        )
        ranks[bucket] += 1
        row["lookup_rank"] = ranks[bucket]
        row["row_fingerprint"] = canonical_semantic_fingerprint(row)
    schema = geographic_reference_neighbours_schema()
    frame = (
        pl.DataFrame(rows, schema=schema, orient="row", strict=True).sort(*_SORT)
        if rows
        else pl.DataFrame(schema=schema)
    )
    validate_geographic_reference_neighbours(frame)
    return frame


def validate_geographic_reference_neighbours(frame: pl.DataFrame) -> None:
    """Reject schema, membership grain, fallback, precision or provenance drift."""

    if not isinstance(frame, pl.DataFrame):
        raise TypeError("geographic reference neighbours must be a Polars DataFrame")
    if frame.schema != geographic_reference_neighbours_schema():
        raise ValueError("geographic reference neighbours schema mismatch")
    if frame.is_empty():
        return
    if frame.select(_GRAIN).n_unique() != frame.height:
        raise ValueError("geographic reference neighbour grain is not unique")
    if frame["membership_id"].n_unique() != frame.height:
        raise ValueError("geographic reference membership IDs are not unique")
    if not frame.equals(frame.sort(*_SORT)):
        raise ValueError("geographic reference neighbours are not canonically sorted")
    for row in frame.iter_rows(named=True):
        _validate_membership_row(row)
    for group in frame.partition_by(
        ["lookup_scope", "lookup_key", "accepted_taxon_key", "route"],
        maintain_order=True,
    ):
        if group["lookup_rank"].to_list() != list(range(1, group.height + 1)):
            raise ValueError("geographic reference lookup ranks are not contiguous")


def geographic_reference_neighbours_artifact_fingerprint(
    frame: pl.DataFrame,
) -> str:
    """Fingerprint lookup semantics independently of physical path and bytes."""

    validate_geographic_reference_neighbours(frame)
    return canonical_semantic_fingerprint(
        {
            "schema_version": GEOGRAPHIC_REFERENCE_NEIGHBOURS_SCHEMA_VERSION,
            "membership_policy_fingerprints": sorted(
                frame["membership_policy_fingerprint"].unique().to_list()
            ),
            "reference_geography_index_fingerprints": sorted(
                frame["reference_geography_index_fingerprint"].unique().to_list()
            ),
            "normalized_reference_geography_fingerprints": sorted(
                frame["normalized_reference_geography_fingerprint"].unique().to_list()
            ),
            "global_reference_anchors_fingerprints": sorted(
                frame["global_reference_anchors_fingerprint"].unique().to_list()
            ),
            "row_fingerprints": sorted(frame["row_fingerprint"].to_list()),
        }
    )


def write_geographic_reference_neighbours(
    frame: pl.DataFrame,
    output: str | Path,
) -> Path:
    """Validate and write the required neighbour lookup artifact."""

    validate_geographic_reference_neighbours(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= GEOGRAPHIC_REFERENCE_NEIGHBOURS_FILE
    return write_parquet(frame, destination)


def _memberships_for_reference(
    index_row: Mapping[str, object],
    *,
    geography: Mapping[str, object],
    is_global_anchor: bool,
    policy: GeographicReferenceNeighbourPolicy,
    grid: CellGrid | None,
    index_fingerprint: str,
    normalized_fingerprint: str,
    anchors_fingerprint: str,
) -> list[dict[str, object]]:
    memberships: list[dict[str, object]] = []
    if index_row["local_anchor_eligible"]:
        if grid is None:
            raise RuntimeError("cell grid was not initialized for local references")
        level, resolution, cell_id = _supported_cell(geography)
        memberships.append(
            _membership(
                index_row,
                geography=geography,
                is_global_anchor=is_global_anchor,
                scope="exact_supported_cell",
                lookup_key=cell_id,
                lookup_cell_level=level,
                lookup_cell_resolution=resolution,
                lookup_cell_id=cell_id,
                neighbour_grid_distance=None,
                policy=policy,
                index_fingerprint=index_fingerprint,
                normalized_fingerprint=normalized_fingerprint,
                anchors_fingerprint=anchors_fingerprint,
            )
        )
        for neighbour in grid.neighbours(
            cell_id,
            grid_distance=policy.neighbour_grid_distance,
            include_origin=False,
        ):
            if neighbour == cell_id:
                raise ValueError("cell grid returned the origin as a neighbour")
            if not grid.is_valid(neighbour):
                raise ValueError("cell grid returned an invalid neighbour")
            memberships.append(
                _membership(
                    index_row,
                    geography=geography,
                    is_global_anchor=is_global_anchor,
                    scope="neighbouring_supported_cell",
                    lookup_key=neighbour,
                    lookup_cell_level=level,
                    lookup_cell_resolution=resolution,
                    lookup_cell_id=neighbour,
                    neighbour_grid_distance=policy.neighbour_grid_distance,
                    policy=policy,
                    index_fingerprint=index_fingerprint,
                    normalized_fingerprint=normalized_fingerprint,
                    anchors_fingerprint=anchors_fingerprint,
                )
            )
        if level == "local":
            memberships.append(
                _parent_membership(
                    index_row,
                    geography=geography,
                    is_global_anchor=is_global_anchor,
                    scope="parent_regional_cell",
                    level="regional",
                    policy=policy,
                    index_fingerprint=index_fingerprint,
                    normalized_fingerprint=normalized_fingerprint,
                    anchors_fingerprint=anchors_fingerprint,
                )
            )
        if level in {"local", "regional"}:
            memberships.append(
                _parent_membership(
                    index_row,
                    geography=geography,
                    is_global_anchor=is_global_anchor,
                    scope="parent_coarse_cell",
                    level="coarse",
                    policy=policy,
                    index_fingerprint=index_fingerprint,
                    normalized_fingerprint=normalized_fingerprint,
                    anchors_fingerprint=anchors_fingerprint,
                )
            )
    for scope, field in (
        ("bioregion", "bioregion"),
        ("country", "country_code"),
        ("continent", "continent_code"),
    ):
        value = geography[field]
        if value is not None:
            memberships.append(
                _membership(
                    index_row,
                    geography=geography,
                    is_global_anchor=is_global_anchor,
                    scope=scope,
                    lookup_key=str(value),
                    lookup_cell_level=None,
                    lookup_cell_resolution=None,
                    lookup_cell_id=None,
                    neighbour_grid_distance=None,
                    policy=policy,
                    index_fingerprint=index_fingerprint,
                    normalized_fingerprint=normalized_fingerprint,
                    anchors_fingerprint=anchors_fingerprint,
                )
            )
    if is_global_anchor:
        memberships.append(
            _membership(
                index_row,
                geography=geography,
                is_global_anchor=True,
                scope="global",
                lookup_key="global",
                lookup_cell_level=None,
                lookup_cell_resolution=None,
                lookup_cell_id=None,
                neighbour_grid_distance=None,
                policy=policy,
                index_fingerprint=index_fingerprint,
                normalized_fingerprint=normalized_fingerprint,
                anchors_fingerprint=anchors_fingerprint,
            )
        )
    return memberships


def _parent_membership(
    index_row: Mapping[str, object],
    *,
    geography: Mapping[str, object],
    is_global_anchor: bool,
    scope: str,
    level: str,
    policy: GeographicReferenceNeighbourPolicy,
    index_fingerprint: str,
    normalized_fingerprint: str,
    anchors_fingerprint: str,
) -> dict[str, object]:
    cell_id = str(geography[f"{level}_cell_id"])
    return _membership(
        index_row,
        geography=geography,
        is_global_anchor=is_global_anchor,
        scope=scope,
        lookup_key=cell_id,
        lookup_cell_level=level,
        lookup_cell_resolution=int(geography[f"{level}_cell_resolution"]),
        lookup_cell_id=cell_id,
        neighbour_grid_distance=None,
        policy=policy,
        index_fingerprint=index_fingerprint,
        normalized_fingerprint=normalized_fingerprint,
        anchors_fingerprint=anchors_fingerprint,
    )


def _membership(
    index_row: Mapping[str, object],
    *,
    geography: Mapping[str, object],
    is_global_anchor: bool,
    scope: str,
    lookup_key: str,
    lookup_cell_level: str | None,
    lookup_cell_resolution: int | None,
    lookup_cell_id: str | None,
    neighbour_grid_distance: int | None,
    policy: GeographicReferenceNeighbourPolicy,
    index_fingerprint: str,
    normalized_fingerprint: str,
    anchors_fingerprint: str,
) -> dict[str, object]:
    supported_level, supported_resolution, supported_cell = _optional_supported_cell(
        geography
    )
    identity = {
        "reference_geography_row_fingerprint": index_row["row_fingerprint"],
        "lookup_scope": scope,
        "lookup_key": lookup_key,
        "membership_policy_fingerprint": policy.fingerprint,
    }
    membership_hash = canonical_semantic_fingerprint(identity).removeprefix("sha256:")
    return {
        "schema_version": GEOGRAPHIC_REFERENCE_NEIGHBOURS_SCHEMA_VERSION,
        "registry_version": index_row["registry_version"],
        "reference_bank_version": index_row["reference_bank_version"],
        "accepted_taxon_key": index_row["accepted_taxon_key"],
        "scientific_name": index_row["scientific_name"],
        "route": index_row["route"],
        "visual_domain": index_row["visual_domain"],
        "reference_media_id": index_row["reference_media_id"],
        "reference_observation_id": index_row["reference_observation_id"],
        "duplicate_group_id": index_row["duplicate_group_id"],
        "source": index_row["source"],
        "embedding_fingerprint": index_row["embedding_fingerprint"],
        "reference_geography_row_fingerprint": index_row["row_fingerprint"],
        "normalized_geography_row_fingerprint": geography["row_fingerprint"],
        "coordinate_quality": geography["coordinate_quality"],
        "geography_unavailable_reason": geography["geography_unavailable_reason"],
        "cell_grid_name": geography["cell_grid_name"],
        "cell_grid_version": geography["cell_grid_version"],
        "coarse_cell_resolution": geography["coarse_cell_resolution"],
        "regional_cell_resolution": geography["regional_cell_resolution"],
        "local_cell_resolution": geography["local_cell_resolution"],
        "supported_cell_level": supported_level,
        "supported_cell_resolution": supported_resolution,
        "supported_cell_id": supported_cell,
        "coarse_cell_id": geography["coarse_cell_id"],
        "regional_cell_id": geography["regional_cell_id"],
        "local_cell_id": geography["local_cell_id"],
        "country_code": geography["country_code"],
        "bioregion": geography["bioregion"],
        "continent_code": geography["continent_code"],
        "is_global_anchor": is_global_anchor,
        "lookup_scope": scope,
        "lookup_key": lookup_key,
        "lookup_cell_level": lookup_cell_level,
        "lookup_cell_resolution": lookup_cell_resolution,
        "lookup_cell_id": lookup_cell_id,
        "neighbour_grid_distance": neighbour_grid_distance,
        "fallback_level": _SCOPE_FALLBACK_LEVEL[scope],
        "inclusion_reason": _SCOPE_REASON[scope],
        "membership_id": f"geographic-reference-neighbour:{membership_hash}",
        "lookup_rank": 0,
        "membership_policy_fingerprint": policy.fingerprint,
        "reference_geography_index_fingerprint": index_fingerprint,
        "normalized_reference_geography_fingerprint": normalized_fingerprint,
        "global_reference_anchors_fingerprint": anchors_fingerprint,
    }


def _supported_cell(geography: Mapping[str, object]) -> tuple[str, int, str]:
    level, resolution, cell_id = _optional_supported_cell(geography)
    if level is None or resolution is None or cell_id is None:
        raise ValueError("local-anchor eligibility lacks a supported cell")
    return level, resolution, cell_id


def _optional_supported_cell(
    geography: Mapping[str, object],
) -> tuple[str | None, int | None, str | None]:
    quality = str(geography["coordinate_quality"])
    if quality not in _GEO_QUALITIES:
        return None, None, None
    return (
        quality,
        int(geography["supported_cell_resolution"]),
        str(geography[f"{quality}_cell_id"]),
    )


def _validated_grid(
    grid: CellGrid | None,
    *,
    cell_rows: pl.DataFrame,
    geography_by_observation: Mapping[str, Mapping[str, object]],
) -> CellGrid | None:
    if cell_rows.is_empty():
        return None
    backend = grid or default_cell_grid()
    for observation_id in cell_rows["reference_observation_id"].unique().to_list():
        geography = geography_by_observation[str(observation_id)]
        if (
            geography["cell_grid_name"] != backend.name
            or geography["cell_grid_version"] != backend.version
        ):
            raise ValueError(
                "normalized geography grid identity differs from neighbour grid"
            )
        _, _, cell_id = _supported_cell(geography)
        if not backend.is_valid(cell_id):
            raise ValueError("normalized geography contains an invalid supported cell")
    return backend


def _require_geography_coverage(
    index: pl.DataFrame,
    *,
    geography_by_observation: Mapping[str, Mapping[str, object]],
) -> None:
    missing = set(index["reference_observation_id"].to_list()) - set(
        geography_by_observation
    )
    if missing:
        raise ValueError(
            "reference index lacks normalized geography for observations: "
            f"{sorted(missing)}"
        )


def _validate_index_geography(
    index_row: Mapping[str, object],
    *,
    geography: Mapping[str, object],
) -> None:
    comparisons = {
        "registry_version": (
            index_row["registry_version"],
            geography["registry_version"],
        ),
        "accepted_taxon_key": (
            index_row["accepted_taxon_key"],
            geography["accepted_taxon_key"],
        ),
        "scientific_name": (index_row["scientific_name"], geography["scientific_name"]),
        "country_code": (index_row["country_code"], geography["country_code"]),
        "bioregion": (index_row["bioregion"], geography["bioregion"]),
        "geo_cluster_id": (
            index_row["geo_cluster_id"],
            geography["source_geo_cluster_id"],
        ),
        "coarse_cell_id": (index_row["coarse_cell_id"], geography["coarse_cell_id"]),
        "regional_cell_id": (
            index_row["regional_cell_id"],
            geography["regional_cell_id"],
        ),
        "local_cell_id": (index_row["local_cell_id"], geography["local_cell_id"]),
        "latitude": (index_row["latitude"], geography["latitude"]),
        "longitude": (index_row["longitude"], geography["longitude"]),
        "coordinate_uncertainty_m": (
            index_row["coordinate_uncertainty_m"],
            geography["coordinate_uncertainty_m"],
        ),
        "coordinate_quality": (
            index_row["coordinate_quality"],
            geography["coordinate_quality"],
        ),
        "observer_id_hash": (
            index_row["observer_id_hash"],
            geography["observer_id_hash"],
        ),
        "observation_date": (
            index_row["observation_date"],
            geography["observed_date"],
        ),
    }
    conflicts = [
        field for field, values in comparisons.items() if values[0] != values[1]
    ]
    if str(index_row["source"]).casefold() != str(geography["source"]).casefold():
        conflicts.append("source")
    if conflicts:
        raise ValueError(
            f"reference index conflicts with normalized geography: {sorted(conflicts)}"
        )


def _validate_anchor_parent(
    anchors: pl.DataFrame,
    *,
    index_fingerprint: str,
    known_index_rows: set[str],
) -> None:
    if anchors.is_empty():
        return
    if anchors["reference_geography_index_fingerprint"].unique().to_list() != [
        index_fingerprint
    ]:
        raise ValueError("global anchors were selected from another reference index")
    unknown = set(anchors["reference_geography_row_fingerprint"].to_list()) - (
        known_index_rows
    )
    if unknown:
        raise ValueError("global anchors contain unknown reference index rows")


def _pre_rank_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["fallback_level"]),
        str(row["lookup_scope"]),
        str(row["lookup_key"]),
        str(row["accepted_taxon_key"]),
        str(row["route"]),
        str(row["reference_observation_id"]),
        str(row["duplicate_group_id"]),
        str(row["reference_media_id"]),
        str(row["embedding_fingerprint"]),
    )


def _validate_membership_row(row: Mapping[str, object]) -> None:
    if row["schema_version"] != GEOGRAPHIC_REFERENCE_NEIGHBOURS_SCHEMA_VERSION:
        raise ValueError("unsupported geographic reference neighbour schema")
    scope = str(row["lookup_scope"])
    if scope not in REFERENCE_NEIGHBOUR_SCOPES:
        raise ValueError("unsupported geographic reference lookup scope")
    if row["fallback_level"] != _SCOPE_FALLBACK_LEVEL[scope]:
        raise ValueError("geographic reference fallback level conflicts with scope")
    if row["inclusion_reason"] != _SCOPE_REASON[scope]:
        raise ValueError("geographic reference inclusion reason conflicts with scope")
    for field in (
        "embedding_fingerprint",
        "reference_geography_row_fingerprint",
        "normalized_geography_row_fingerprint",
        "membership_policy_fingerprint",
        "reference_geography_index_fingerprint",
        "normalized_reference_geography_fingerprint",
        "global_reference_anchors_fingerprint",
        "row_fingerprint",
    ):
        if not _SHA256_PATTERN.fullmatch(str(row[field])):
            raise ValueError(f"{field} is not a canonical SHA-256 fingerprint")
    if not _MEMBERSHIP_ID_PATTERN.fullmatch(str(row["membership_id"])):
        raise ValueError("geographic reference membership_id is invalid")
    if scope in _CELL_SCOPES:
        _validate_cell_membership(row, scope=scope)
    elif scope in _NAMED_SCOPES:
        if any(
            row[field] is not None
            for field in (
                "lookup_cell_level",
                "lookup_cell_resolution",
                "lookup_cell_id",
                "neighbour_grid_distance",
            )
        ):
            raise ValueError("named fallback scope cannot carry lookup cell claims")
        expected_key = (
            "global"
            if scope == "global"
            else row[f"{scope}_code"]
            if scope in {"country", "continent"}
            else row["bioregion"]
        )
        if row["lookup_key"] != expected_key:
            raise ValueError(
                "named fallback lookup key conflicts with source geography"
            )
        if scope == "global" and not row["is_global_anchor"]:
            raise ValueError("global fallback requires a selected global anchor")
    identity = {
        "reference_geography_row_fingerprint": row[
            "reference_geography_row_fingerprint"
        ],
        "lookup_scope": scope,
        "lookup_key": row["lookup_key"],
        "membership_policy_fingerprint": row["membership_policy_fingerprint"],
    }
    expected_id = canonical_semantic_fingerprint(identity).removeprefix("sha256:")
    if row["membership_id"] != f"geographic-reference-neighbour:{expected_id}":
        raise ValueError("geographic reference membership identity drifted")
    payload = dict(row)
    fingerprint = payload.pop("row_fingerprint")
    if fingerprint != canonical_semantic_fingerprint(payload):
        raise ValueError("geographic reference neighbour row fingerprint mismatch")


def _validate_cell_membership(row: Mapping[str, object], *, scope: str) -> None:
    if row["coordinate_quality"] not in _GEO_QUALITIES:
        raise ValueError("cell fallback cannot use unsupported coordinate precision")
    if any(
        row[field] is None
        for field in (
            "supported_cell_level",
            "supported_cell_resolution",
            "supported_cell_id",
            "lookup_cell_level",
            "lookup_cell_resolution",
            "lookup_cell_id",
        )
    ):
        raise ValueError("cell fallback requires explicit supported and lookup cells")
    if row["lookup_key"] != row["lookup_cell_id"]:
        raise ValueError("cell fallback lookup key differs from lookup cell")
    if scope == "exact_supported_cell":
        if (
            row["lookup_cell_level"] != row["supported_cell_level"]
            or row["lookup_cell_resolution"] != row["supported_cell_resolution"]
            or row["lookup_cell_id"] != row["supported_cell_id"]
            or row["neighbour_grid_distance"] is not None
        ):
            raise ValueError("exact fallback differs from the supported reference cell")
    elif scope == "neighbouring_supported_cell":
        if (
            row["lookup_cell_level"] != row["supported_cell_level"]
            or row["lookup_cell_resolution"] != row["supported_cell_resolution"]
            or row["lookup_cell_id"] == row["supported_cell_id"]
            or row["neighbour_grid_distance"] != 1
        ):
            raise ValueError("neighbour fallback differs from supported-cell precision")
    elif scope == "parent_regional_cell":
        if (
            row["supported_cell_level"] != "local"
            or row["lookup_cell_level"] != "regional"
            or row["lookup_cell_id"] != row["regional_cell_id"]
            or row["lookup_cell_resolution"] != row["regional_cell_resolution"]
            or row["neighbour_grid_distance"] is not None
        ):
            raise ValueError("regional parent fallback is inconsistent")
    elif (
        row["supported_cell_level"] not in {"local", "regional"}
        or row["lookup_cell_level"] != "coarse"
        or row["lookup_cell_id"] != row["coarse_cell_id"]
        or row["lookup_cell_resolution"] != row["coarse_cell_resolution"]
        or row["neighbour_grid_distance"] is not None
    ):
        raise ValueError("coarse parent fallback is inconsistent")


__all__ = [
    "GEOGRAPHIC_REFERENCE_NEIGHBOURS_FILE",
    "GEOGRAPHIC_REFERENCE_NEIGHBOURS_SCHEMA_VERSION",
    "GEOGRAPHIC_REFERENCE_NEIGHBOUR_POLICY_VERSION",
    "REFERENCE_NEIGHBOUR_SCOPES",
    "GeographicReferenceNeighbourPolicy",
    "build_geographic_reference_neighbours",
    "geographic_reference_neighbours_artifact_fingerprint",
    "geographic_reference_neighbours_schema",
    "validate_geographic_reference_neighbours",
    "write_geographic_reference_neighbours",
]
