"""Deterministic geography/taxon partitions over canonical Flickr scoring units."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.flickr_fetch.scoring_geography import (
    validate_flickr_scoring_geography,
)
from biominer.flickr_fetch.scoring_units import (
    FlickrScoringUnitArtifacts,
    validate_flickr_scoring_unit_artifacts,
)
from biominer.storage.parquet import write_parquet


FLICKR_GEO_TAXON_PARTITION_SCHEMA_VERSION = "flickr-geo-taxon-partition-v1.0.0"
FLICKR_PARTITION_SUMMARY_SCHEMA_VERSION = "flickr-partition-summary-v1.0.0"
FLICKR_PARTITION_POLICY_VERSION = "flickr-geo-taxon-partition-policy-v1.0.0"
FLICKR_GEO_TAXON_PARTITIONS_FILE = "flickr_geo_taxon_partitions.parquet"
FLICKR_PARTITION_SUMMARY_FILE = "flickr_partition_summary.parquet"

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PARTITION_ID_PATTERN = re.compile(r"flickr-geo-taxon-partition:[0-9a-f]{64}\Z")
_PARTITION_SORT = (
    "run_id",
    "partition_id",
    "source",
    "flickr_photo_id",
    "route",
    "organism_unit_id",
)
_SUMMARY_SORT = ("run_id", "partition_id")


@dataclass(frozen=True, slots=True)
class FlickrPartitionArtifacts:
    partitions: pl.DataFrame
    summary: pl.DataFrame


def flickr_geo_taxon_partition_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "partition_id": pl.String,
        "run_id": pl.String,
        "organism_unit_id": pl.String,
        "photo_embedding_unit_id": pl.String,
        "source": pl.String,
        "flickr_photo_id": pl.String,
        "route": pl.String,
        "visual_input_id": pl.String,
        "model_input_signature": pl.String,
        "model_input_contract_signature": pl.String,
        "geography_availability": pl.String,
        "geographic_scope": pl.String,
        "geographic_scope_value": pl.String,
        "source_geography_signature": pl.String,
        "geographic_work_signature": pl.String,
        "candidate_set_signature": pl.String,
        "candidate_species_count": pl.UInt32,
        "family_pool_signature": pl.String,
        "family_count": pl.UInt32,
        "association_set_signature": pl.String,
        "association_count": pl.UInt32,
        "partition_policy_version": pl.String,
        "partition_policy_fingerprint": pl.String,
        "row_fingerprint": pl.String,
    }


def flickr_partition_summary_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "partition_id": pl.String,
        "run_id": pl.String,
        "route": pl.String,
        "geography_availability": pl.String,
        "geographic_scope": pl.String,
        "geographic_scope_value": pl.String,
        "geographic_work_signature": pl.String,
        "candidate_set_signature": pl.String,
        "candidate_species_count": pl.UInt32,
        "family_pool_signature": pl.String,
        "family_count": pl.UInt32,
        "model_input_contract_signature": pl.String,
        "organism_unit_count": pl.UInt64,
        "photo_embedding_unit_count": pl.UInt64,
        "visual_input_count": pl.UInt64,
        "model_input_count": pl.UInt64,
        "model_input_reuse_count": pl.UInt64,
        "association_set_count": pl.UInt64,
        "association_count": pl.UInt64,
        "partition_policy_version": pl.String,
        "partition_policy_fingerprint": pl.String,
        "partition_membership_fingerprint": pl.String,
        "summary_fingerprint": pl.String,
    }


def build_flickr_geo_taxon_partitions(
    scoring_artifacts: FlickrScoringUnitArtifacts,
    scoring_geography: pl.DataFrame,
    *,
    partition_policy_version: str = FLICKR_PARTITION_POLICY_VERSION,
) -> FlickrPartitionArtifacts:
    """Assign every organism unit once without materializing images or vectors."""

    validate_flickr_scoring_unit_artifacts(scoring_artifacts)
    validate_flickr_scoring_geography(
        scoring_geography, scoring_artifacts.photo_embedding_units
    )
    policy_version = _required_text(
        partition_policy_version, field="partition_policy_version"
    )
    policy_fingerprint = _partition_policy_fingerprint(policy_version)
    partition_rows = _build_partition_rows(
        scoring_artifacts,
        scoring_geography,
        policy_version=policy_version,
        policy_fingerprint=policy_fingerprint,
    )
    partitions = _frame(
        partition_rows,
        schema=flickr_geo_taxon_partition_schema(),
        sort=_PARTITION_SORT,
    )
    summary = _frame(
        _build_summary_rows(partitions),
        schema=flickr_partition_summary_schema(),
        sort=_SUMMARY_SORT,
    )
    result = FlickrPartitionArtifacts(partitions=partitions, summary=summary)
    validate_flickr_geo_taxon_partitions(
        result,
        scoring_artifacts,
        scoring_geography,
    )
    return result


def validate_flickr_geo_taxon_partitions(
    artifacts: FlickrPartitionArtifacts,
    scoring_artifacts: FlickrScoringUnitArtifacts,
    scoring_geography: pl.DataFrame,
) -> None:
    if not isinstance(artifacts, FlickrPartitionArtifacts):
        raise TypeError("artifacts must be FlickrPartitionArtifacts")
    validate_flickr_scoring_unit_artifacts(scoring_artifacts)
    validate_flickr_scoring_geography(
        scoring_geography, scoring_artifacts.photo_embedding_units
    )
    _validate_frame(
        artifacts.partitions,
        schema=flickr_geo_taxon_partition_schema(),
        sort=_PARTITION_SORT,
        label="Flickr geo/taxon partitions",
    )
    _validate_frame(
        artifacts.summary,
        schema=flickr_partition_summary_schema(),
        sort=_SUMMARY_SORT,
        label="Flickr partition summary",
    )
    partitions = artifacts.partitions
    summary = artifacts.summary
    if partitions.height != partitions["organism_unit_id"].n_unique():
        raise ValueError("every organism unit must have exactly one partition row")
    if partitions.height != scoring_artifacts.scoring_units.height:
        raise ValueError("partition rows do not cover every organism unit")
    if summary.height != summary["partition_id"].n_unique():
        raise ValueError("partition summary grain is not unique")
    if set(partitions["partition_id"].to_list()) != set(
        summary["partition_id"].to_list()
    ):
        raise ValueError("partition assignments and summary identities differ")

    policy_versions = set(partitions["partition_policy_version"].to_list())
    if not partitions.is_empty() and len(policy_versions) != 1:
        raise ValueError("partition rows require one policy version")
    policy_version = (
        next(iter(policy_versions))
        if policy_versions
        else FLICKR_PARTITION_POLICY_VERSION
    )
    policy_fingerprint = _partition_policy_fingerprint(policy_version)
    expected_partitions = _frame(
        _build_partition_rows(
            scoring_artifacts,
            scoring_geography,
            policy_version=policy_version,
            policy_fingerprint=policy_fingerprint,
        ),
        schema=flickr_geo_taxon_partition_schema(),
        sort=_PARTITION_SORT,
    )
    if not partitions.equals(expected_partitions):
        raise ValueError("Flickr partition assignments do not match source contracts")
    expected_summary = _frame(
        _build_summary_rows(partitions),
        schema=flickr_partition_summary_schema(),
        sort=_SUMMARY_SORT,
    )
    if not summary.equals(expected_summary):
        raise ValueError("Flickr partition summary does not match assignments")
    for row in partitions.iter_rows(named=True):
        if row["schema_version"] != FLICKR_GEO_TAXON_PARTITION_SCHEMA_VERSION:
            raise ValueError("unsupported Flickr geo/taxon partition schema version")
        if not _PARTITION_ID_PATTERN.fullmatch(str(row["partition_id"])):
            raise ValueError("invalid Flickr geo/taxon partition ID")
        for field in (
            "organism_unit_id",
            "visual_input_id",
            "model_input_signature",
            "model_input_contract_signature",
            "source_geography_signature",
            "geographic_work_signature",
            "candidate_set_signature",
            "family_pool_signature",
            "association_set_signature",
            "partition_policy_fingerprint",
            "row_fingerprint",
        ):
            _sha256(row[field], field=field)


def write_flickr_partition_artifacts(
    artifacts: FlickrPartitionArtifacts,
    scoring_artifacts: FlickrScoringUnitArtifacts,
    scoring_geography: pl.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    validate_flickr_geo_taxon_partitions(
        artifacts,
        scoring_artifacts,
        scoring_geography,
    )
    output = Path(output_dir)
    return {
        "partitions": write_parquet(
            artifacts.partitions,
            output / FLICKR_GEO_TAXON_PARTITIONS_FILE,
        ),
        "summary": write_parquet(
            artifacts.summary,
            output / FLICKR_PARTITION_SUMMARY_FILE,
        ),
    }


def _build_partition_rows(
    scoring_artifacts: FlickrScoringUnitArtifacts,
    scoring_geography: pl.DataFrame,
    *,
    policy_version: str,
    policy_fingerprint: str,
) -> list[dict[str, object]]:
    photo_by_id = {
        str(row["photo_embedding_unit_id"]): row
        for row in scoring_artifacts.photo_embedding_units.iter_rows(named=True)
    }
    geography_by_photo = {
        str(row["photo_embedding_unit_id"]): row
        for row in scoring_geography.iter_rows(named=True)
    }
    candidates_by_unit = _rows_by_unit(scoring_artifacts.candidate_species)
    associations_by_unit = _rows_by_unit(scoring_artifacts.associations)
    output: list[dict[str, object]] = []
    for unit in scoring_artifacts.scoring_units.iter_rows(named=True):
        organism_unit_id = str(unit["organism_unit_id"])
        photo_id = str(unit["photo_embedding_unit_id"])
        photo = photo_by_id[photo_id]
        geography = geography_by_photo[photo_id]
        candidates = candidates_by_unit.get(organism_unit_id, [])
        associations = associations_by_unit.get(organism_unit_id, [])
        model_contract_signature = _model_input_contract_signature(photo)
        geographic_work_signature = _geographic_work_signature(geography)
        candidate_signature = _candidate_set_signature(candidates)
        family_signature = _family_pool_signature(candidates)
        association_signature = _association_set_signature(associations)
        families = {
            (str(row["family_key"]), str(row["family_name"]))
            for row in candidates
        }
        partition_context = {
            "schema_version": "flickr-geo-taxon-partition-id-v1",
            "run_id": unit["run_id"],
            "route": unit["route"],
            "geographic_work_signature": geographic_work_signature,
            "candidate_set_signature": candidate_signature,
            "family_pool_signature": family_signature,
            "model_input_contract_signature": model_contract_signature,
            "partition_policy_fingerprint": policy_fingerprint,
        }
        partition_fingerprint = canonical_semantic_fingerprint(partition_context)
        base = {
            "schema_version": FLICKR_GEO_TAXON_PARTITION_SCHEMA_VERSION,
            "partition_id": _prefixed_id(
                "flickr-geo-taxon-partition", partition_fingerprint
            ),
            "run_id": unit["run_id"],
            "organism_unit_id": organism_unit_id,
            "photo_embedding_unit_id": photo_id,
            "source": unit["source"],
            "flickr_photo_id": unit["flickr_photo_id"],
            "route": unit["route"],
            "visual_input_id": unit["visual_input_id"],
            "model_input_signature": unit["model_input_signature"],
            "model_input_contract_signature": model_contract_signature,
            "geography_availability": geography["geography_availability"],
            "geographic_scope": geography["geographic_scope"],
            "geographic_scope_value": geography["geographic_scope_value"],
            "source_geography_signature": geography["geography_signature"],
            "geographic_work_signature": geographic_work_signature,
            "candidate_set_signature": candidate_signature,
            "candidate_species_count": len(candidates),
            "family_pool_signature": family_signature,
            "family_count": len(families),
            "association_set_signature": association_signature,
            "association_count": len(associations),
            "partition_policy_version": policy_version,
            "partition_policy_fingerprint": policy_fingerprint,
        }
        output.append(
            {**base, "row_fingerprint": canonical_semantic_fingerprint(base)}
        )
    return output


def _build_summary_rows(partitions: pl.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for partition_id in sorted(partitions["partition_id"].unique().to_list()):
        members = partitions.filter(pl.col("partition_id") == partition_id).sort(
            "organism_unit_id"
        )
        first = members.row(0, named=True)
        membership_fingerprint = canonical_semantic_fingerprint(
            {
                "schema_version": "flickr-partition-membership-v1",
                "partition_id": partition_id,
                "member_row_fingerprints": members["row_fingerprint"].to_list(),
            }
        )
        organism_count = members.height
        model_input_count = members["model_input_signature"].n_unique()
        base = {
            "schema_version": FLICKR_PARTITION_SUMMARY_SCHEMA_VERSION,
            "partition_id": partition_id,
            "run_id": first["run_id"],
            "route": first["route"],
            "geography_availability": first["geography_availability"],
            "geographic_scope": first["geographic_scope"],
            "geographic_scope_value": first["geographic_scope_value"],
            "geographic_work_signature": first["geographic_work_signature"],
            "candidate_set_signature": first["candidate_set_signature"],
            "candidate_species_count": first["candidate_species_count"],
            "family_pool_signature": first["family_pool_signature"],
            "family_count": first["family_count"],
            "model_input_contract_signature": first[
                "model_input_contract_signature"
            ],
            "organism_unit_count": organism_count,
            "photo_embedding_unit_count": members[
                "photo_embedding_unit_id"
            ].n_unique(),
            "visual_input_count": members["visual_input_id"].n_unique(),
            "model_input_count": model_input_count,
            "model_input_reuse_count": organism_count - model_input_count,
            "association_set_count": members[
                "association_set_signature"
            ].n_unique(),
            "association_count": int(members["association_count"].sum()),
            "partition_policy_version": first["partition_policy_version"],
            "partition_policy_fingerprint": first[
                "partition_policy_fingerprint"
            ],
            "partition_membership_fingerprint": membership_fingerprint,
        }
        rows.append(
            {**base, "summary_fingerprint": canonical_semantic_fingerprint(base)}
        )
    return rows


def _rows_by_unit(frame: pl.DataFrame) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in frame.iter_rows(named=True):
        grouped.setdefault(str(row["organism_unit_id"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: str(row.get("candidate_species_id") or row.get("association_id")))
    return grouped


def _model_input_contract_signature(photo: Mapping[str, object]) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": "flickr-model-input-contract-signature-v1",
            "visual_input_kind": photo["visual_input_kind"],
            "visual_input_version": photo["visual_input_version"],
            "transformation_fingerprint": photo["transformation_fingerprint"],
            "image_resize_mode": photo["image_resize_mode"],
            "preprocessing_contract_fingerprint": photo[
                "preprocessing_contract_fingerprint"
            ],
            "image_mode": photo["image_mode"],
        }
    )


def _geographic_work_signature(geography: Mapping[str, object]) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": "flickr-geographic-work-signature-v1",
            "geography_availability": geography["geography_availability"],
            "geographic_scope": geography["geographic_scope"],
            "geographic_scope_value": geography["geographic_scope_value"],
            "coordinate_quality": geography["coordinate_quality"],
            "supported_cell_resolution": geography["supported_cell_resolution"],
            "geography_policy_fingerprint": geography[
                "geography_policy_fingerprint"
            ],
        }
    )


def _candidate_set_signature(candidates: Sequence[Mapping[str, object]]) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": "flickr-partition-candidate-set-v1",
            "candidates": sorted(
                (
                    {
                        "accepted_taxon_key": row[
                            "candidate_accepted_taxon_key"
                        ],
                        "scientific_name": row["candidate_scientific_name"],
                        "family_key": row["family_key"],
                        "genus_key": row["genus_key"],
                    }
                    for row in candidates
                ),
                key=lambda value: str(value["accepted_taxon_key"]),
            ),
        }
    )


def _family_pool_signature(candidates: Sequence[Mapping[str, object]]) -> str:
    families = sorted(
        {
            (str(row["family_key"]), str(row["family_name"]))
            for row in candidates
        }
    )
    return canonical_semantic_fingerprint(
        {
            "schema_version": "flickr-partition-family-pool-v1",
            "families": [
                {"family_key": key, "family_name": name} for key, name in families
            ],
        }
    )


def _association_set_signature(
    associations: Sequence[Mapping[str, object]],
) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": "flickr-partition-association-set-v1",
            "associations": [
                {
                    "association_kind": row["association_kind"],
                    "association_source": row["association_source"],
                    "association_source_id": row["association_source_id"],
                    "flickr_query_id": row["flickr_query_id"],
                    "query_hash": row["query_hash"],
                    "accepted_taxon_key": row["accepted_taxon_key"],
                }
                for row in sorted(
                    associations, key=lambda value: str(value["association_id"])
                )
            ],
        }
    )


def _partition_policy_fingerprint(version: str) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": FLICKR_GEO_TAXON_PARTITION_SCHEMA_VERSION,
            "partition_policy_version": version,
            "partition_keys": [
                "run_id",
                "route",
                "geographic_work_signature",
                "candidate_set_signature",
                "family_pool_signature",
                "model_input_contract_signature",
            ],
            "content_identity_role": "cache_reference_not_partition_key",
            "family_membership_role": "batching_only_no_candidate_pruning",
        }
    )


def _frame(
    rows: Sequence[Mapping[str, object]],
    *,
    schema: dict[str, pl.DataType],
    sort: Sequence[str],
) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row", strict=True).sort(*sort)


def _validate_frame(
    frame: pl.DataFrame,
    *,
    schema: dict[str, pl.DataType],
    sort: Sequence[str],
    label: str,
) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{label} must be a Polars DataFrame")
    if frame.schema != schema:
        raise ValueError(f"{label} schema mismatch")
    if not frame.equals(frame.sort(*sort)):
        raise ValueError(f"{label} is not canonically sorted")


def _prefixed_id(prefix: str, fingerprint: str) -> str:
    return f"{prefix}:{fingerprint.removeprefix('sha256:')}"


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase sha256 fingerprint")
    return text


__all__ = [
    "FLICKR_GEO_TAXON_PARTITIONS_FILE",
    "FLICKR_GEO_TAXON_PARTITION_SCHEMA_VERSION",
    "FLICKR_PARTITION_POLICY_VERSION",
    "FLICKR_PARTITION_SUMMARY_FILE",
    "FLICKR_PARTITION_SUMMARY_SCHEMA_VERSION",
    "FlickrPartitionArtifacts",
    "build_flickr_geo_taxon_partitions",
    "flickr_geo_taxon_partition_schema",
    "flickr_partition_summary_schema",
    "validate_flickr_geo_taxon_partitions",
    "write_flickr_partition_artifacts",
]
