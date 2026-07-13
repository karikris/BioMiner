from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import struct

import polars as pl

from biominer.candidates.regional_occurrence import (
    REGIONAL_TAXON_OCCURRENCE_SCHEMA_VERSION,
)
from biominer.flickr_fetch.geographic_clustering import NO_GEO_CLUSTER_ID
from biominer.storage.parquet import write_parquet


REGIONAL_CANDIDATE_SPECIES_SCHEMA_VERSION = "regional-candidate-species-v1.0.0"
REGIONAL_CANDIDATE_POLICY_VERSION = "regional-candidate-union-v1.0.0"
GEOGRAPHIC_EVIDENCE_SCORE_VERSION = "overlap-weighted-coordinate-evidence-v1.0.0"
REGIONAL_CANDIDATE_SPECIES_FILE = "regional_candidate_species.parquet"

_OVERLAP_PRIORITY = {
    "exact": 0,
    "buffer": 1,
    "country": 2,
    "bioregion": 3,
    "global": 4,
}
_OVERLAP_WEIGHT = {
    "exact": 1.0,
    "buffer": 0.8,
    "country": 0.5,
    "bioregion": 0.35,
    "global": 0.1,
}
_REASON_PRIORITY = {
    "target": 0,
    "known_mimic": 10,
    "historical_false_positive": 20,
    "visually_nearest": 30,
    "close_congener": 40,
    "same_genus_range_overlap": 50,
    "regional_same_family": 60,
    "taxonomic_neighbour": 70,
    "country_fallback": 80,
    "bioregion_fallback": 90,
    "global_no_geo_fallback": 100,
    "global_registry_fallback": 110,
}
_RELATIONSHIP_REASON = {
    "known_mimic": "known_mimic",
    "close_congener": "close_congener",
    "historical_false_positive_species": "historical_false_positive",
    "historical_false_positive_genus": "historical_false_positive",
    "taxonomic_neighbour": "taxonomic_neighbour",
    "visual_neighbour": "visually_nearest",
}


@dataclass(frozen=True, slots=True)
class RegionalCandidateConfig:
    minimum_local_same_family_candidates: int = 20
    include_registry_same_family_for_no_geo: bool = True
    policy_version: str = REGIONAL_CANDIDATE_POLICY_VERSION

    def __post_init__(self) -> None:
        minimum = self.minimum_local_same_family_candidates
        if isinstance(minimum, bool) or not isinstance(minimum, int):
            raise TypeError("minimum_local_same_family_candidates must be an integer")
        if minimum < 0:
            raise ValueError("minimum_local_same_family_candidates must be non-negative")
        if not isinstance(self.include_registry_same_family_for_no_geo, bool):
            raise TypeError("include_registry_same_family_for_no_geo must be boolean")
        policy = _required_text(self.policy_version, field="policy_version")
        object.__setattr__(self, "policy_version", policy)


@dataclass(frozen=True, slots=True)
class _Taxon:
    accepted_taxon_key: str
    scientific_name: str
    family: str
    genus: str


@dataclass(frozen=True, slots=True)
class _GeoEvidence:
    overlap_type: str
    occurrence_support: int
    geographic_evidence_score: float | None
    evidence_versions: tuple[str, ...]


@dataclass(slots=True)
class _Candidate:
    taxon: _Taxon
    reasons: set[str] = field(default_factory=set)
    known_mimic: bool = False
    historical_false_positive: bool = False
    visually_nearest: bool = False


def regional_candidate_species_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "candidate_set_id": pl.String,
        "target_accepted_taxon_key": pl.String,
        "geo_cluster_id": pl.String,
        "candidate_accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "family": pl.String,
        "genus": pl.String,
        "candidate_reason": pl.List(pl.String),
        "geographic_evidence_score": pl.Float32,
        "occurrence_support": pl.UInt64,
        "same_genus": pl.Boolean,
        "same_family": pl.Boolean,
        "known_mimic": pl.Boolean,
        "historical_false_positive": pl.Boolean,
        "visually_nearest": pl.Boolean,
        "target_candidate": pl.Boolean,
        "candidate_priority": pl.UInt32,
        "source_versions": pl.List(pl.String),
        "candidate_set_fingerprint": pl.String,
    }


def build_regional_candidate_species(
    *,
    target_accepted_taxon_key: str,
    geo_clusters: pl.DataFrame,
    regional_occurrence: pl.DataFrame,
    taxa: pl.DataFrame,
    registry_version: str,
    competitor_relationships: pl.DataFrame | None = None,
    historical_false_positive_taxon_keys: Sequence[str] = (),
    historical_false_positive_version: str | None = None,
    visually_nearest_taxon_keys: Sequence[str] = (),
    visual_neighbour_version: str | None = None,
    source_versions: Sequence[str] = (),
    config: RegionalCandidateConfig | None = None,
) -> pl.DataFrame:
    if not isinstance(geo_clusters, pl.DataFrame):
        raise TypeError("geo_clusters must be a Polars DataFrame")
    if not isinstance(regional_occurrence, pl.DataFrame):
        raise TypeError("regional_occurrence must be a Polars DataFrame")
    if not isinstance(taxa, pl.DataFrame):
        raise TypeError("taxa must be a Polars DataFrame")
    target_key = _required_text(
        target_accepted_taxon_key,
        field="target_accepted_taxon_key",
    )
    registry = _required_text(registry_version, field="registry_version")
    effective_config = config or RegionalCandidateConfig()
    if not isinstance(effective_config, RegionalCandidateConfig):
        raise TypeError("config must be a RegionalCandidateConfig")
    taxonomy = _accepted_species_taxonomy(taxa)
    if target_key not in taxonomy:
        raise ValueError(f"target is not an accepted species in the registry: {target_key}")
    target = taxonomy[target_key]
    clusters = _validated_clusters(geo_clusters, target_key=target_key)
    geography = _regional_geography(
        regional_occurrence,
        taxonomy=taxonomy,
        registry_version=registry,
        cluster_ids=set(clusters),
    )
    relationship_reasons, relationship_versions = _relationship_candidates(
        competitor_relationships,
        target_key=target_key,
        taxonomy=taxonomy,
    )
    false_positive_keys = _accepted_candidate_keys(
        historical_false_positive_taxon_keys,
        taxonomy=taxonomy,
        field="historical_false_positive_taxon_keys",
    )
    visual_keys = _accepted_candidate_keys(
        visually_nearest_taxon_keys,
        taxonomy=taxonomy,
        field="visually_nearest_taxon_keys",
    )
    caller_versions = _versions(source_versions, field="source_versions")
    false_positive_version = _required_dependency_version(
        false_positive_keys,
        historical_false_positive_version,
        field="historical_false_positive_version",
        prefix="false-positives",
    )
    visual_version = _required_dependency_version(
        visual_keys,
        visual_neighbour_version,
        field="visual_neighbour_version",
        prefix="visual-neighbours",
    )

    output_rows: list[dict[str, object]] = []
    for cluster_id in sorted(clusters):
        cluster_geo = geography.get(cluster_id, {})
        candidates: dict[str, _Candidate] = {}

        def include(
            taxon_key: str,
            reason: str,
            *,
            known_mimic: bool = False,
            historical_false_positive: bool = False,
            visually_nearest: bool = False,
        ) -> None:
            candidate = candidates.setdefault(taxon_key, _Candidate(taxonomy[taxon_key]))
            candidate.reasons.add(reason)
            candidate.known_mimic |= known_mimic
            candidate.historical_false_positive |= historical_false_positive
            candidate.visually_nearest |= visually_nearest

        include(target_key, "target")
        local_same_family = {
            key
            for key, evidence in cluster_geo.items()
            if key != target_key
            and taxonomy[key].family == target.family
            and evidence.overlap_type in {"exact", "buffer"}
        }
        sparse_local = (
            len(local_same_family)
            < effective_config.minimum_local_same_family_candidates
        )
        for key, evidence in sorted(cluster_geo.items()):
            taxon = taxonomy[key]
            same_genus = taxon.genus == target.genus
            same_family = taxon.family == target.family
            if key != target_key and same_genus and (
                evidence.overlap_type != "global" or cluster_id == NO_GEO_CLUSTER_ID
            ):
                include(key, "same_genus_range_overlap")
            if key != target_key and same_family and evidence.overlap_type in {
                "exact",
                "buffer",
            }:
                include(key, "regional_same_family")
            if sparse_local and key != target_key and same_family:
                if evidence.overlap_type == "country":
                    include(key, "country_fallback")
                elif evidence.overlap_type == "bioregion":
                    include(key, "bioregion_fallback")
            if (
                cluster_id == NO_GEO_CLUSTER_ID
                and key != target_key
                and same_family
                and evidence.overlap_type == "global"
            ):
                include(key, "global_no_geo_fallback")

        if (
            cluster_id == NO_GEO_CLUSTER_ID
            and effective_config.include_registry_same_family_for_no_geo
        ):
            for key, taxon in sorted(taxonomy.items()):
                if taxon.family == target.family:
                    include(key, "global_registry_fallback")

        for key, reasons in sorted(relationship_reasons.items()):
            for reason in reasons:
                include(
                    key,
                    reason,
                    known_mimic=reason == "known_mimic",
                    historical_false_positive=reason == "historical_false_positive",
                    visually_nearest=reason == "visually_nearest",
                )
        for key in false_positive_keys:
            include(key, "historical_false_positive", historical_false_positive=True)
        for key in visual_keys:
            include(key, "visually_nearest", visually_nearest=True)

        dependency_versions = {
            f"registry:{registry}",
            f"candidate-policy:{effective_config.policy_version}",
            (
                "candidate-config:"
                f"min-local={effective_config.minimum_local_same_family_candidates};"
                "registry-no-geo="
                f"{str(effective_config.include_registry_same_family_for_no_geo).lower()}"
            ),
            f"geographic-score:{GEOGRAPHIC_EVIDENCE_SCORE_VERSION}",
            *caller_versions,
            *relationship_versions,
        }
        if false_positive_version:
            dependency_versions.add(false_positive_version)
        if visual_version:
            dependency_versions.add(visual_version)
        cluster_hash = clusters[cluster_id]
        if cluster_hash:
            dependency_versions.add(f"flickr-clusters:{cluster_hash}")
        for evidence in cluster_geo.values():
            dependency_versions.update(
                f"regional-occurrence:{version}" for version in evidence.evidence_versions
            )
        versions = sorted(dependency_versions)

        ordered = sorted(
            candidates.values(),
            key=lambda candidate: _candidate_order_key(
                candidate,
                target=target,
                geography=cluster_geo.get(candidate.taxon.accepted_taxon_key),
            ),
        )
        provisional_rows = [
            _candidate_row(
                candidate,
                target=target,
                target_key=target_key,
                cluster_id=cluster_id,
                geography=cluster_geo.get(candidate.taxon.accepted_taxon_key),
                candidate_priority=priority,
                source_versions=versions,
            )
            for priority, candidate in enumerate(ordered)
        ]
        fingerprint = _candidate_set_fingerprint(
            target_key=target_key,
            cluster_id=cluster_id,
            config=effective_config,
            source_versions=versions,
            candidates=provisional_rows,
        )
        candidate_set_id = f"regional:{fingerprint.removeprefix('sha256:')[:32]}"
        for row in provisional_rows:
            row["candidate_set_id"] = candidate_set_id
            row["candidate_set_fingerprint"] = fingerprint
            output_rows.append(row)
    return _candidate_frame(output_rows)


def write_regional_candidate_species(
    frame: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    validate_regional_candidate_species(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= REGIONAL_CANDIDATE_SPECIES_FILE
    return write_parquet(frame, destination, overwrite=overwrite)


def validate_regional_candidate_species(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a Polars DataFrame")
    if frame.schema != regional_candidate_species_schema():
        raise ValueError("regional candidate species frame does not match the physical schema")
    expected = frame.sort(
        ["candidate_set_id", "candidate_priority", "candidate_accepted_taxon_key"]
    )
    if not frame.equals(expected):
        raise ValueError("regional candidate species frame is not in deterministic sort order")
    cluster_sets: dict[tuple[str, str], str] = {}
    for (candidate_set_id,), group in frame.group_by("candidate_set_id"):
        if set(group["schema_version"].to_list()) != {
            REGIONAL_CANDIDATE_SPECIES_SCHEMA_VERSION
        }:
            raise ValueError("unsupported regional candidate species schema version")
        priorities = group.sort("candidate_priority")["candidate_priority"].to_list()
        if priorities != list(range(group.height)):
            raise ValueError(f"candidate set {candidate_set_id!r} has invalid priorities")
        if group["target_candidate"].sum() != 1:
            raise ValueError(f"candidate set {candidate_set_id!r} must contain one target")
        if group["candidate_accepted_taxon_key"].n_unique() != group.height:
            raise ValueError(f"candidate set {candidate_set_id!r} contains duplicate species")
        if group["candidate_set_fingerprint"].n_unique() != 1:
            raise ValueError(f"candidate set {candidate_set_id!r} has conflicting fingerprints")
        target_keys = set(group["target_accepted_taxon_key"].to_list())
        if len(target_keys) != 1:
            raise ValueError(f"candidate set {candidate_set_id!r} has conflicting targets")
        target_key = str(next(iter(target_keys)))
        target_row = group.filter(pl.col("target_candidate"))
        if target_row["candidate_accepted_taxon_key"].item() != target_key:
            raise ValueError(
                f"candidate set {candidate_set_id!r} target flag does not identify its target"
            )
        cluster_ids = set(group["geo_cluster_id"].to_list())
        if len(cluster_ids) != 1:
            raise ValueError(f"candidate set {candidate_set_id!r} spans multiple clusters")
        cluster_id = str(next(iter(cluster_ids)))
        cluster_key = (target_key, cluster_id)
        previous = cluster_sets.setdefault(cluster_key, str(candidate_set_id))
        if previous != str(candidate_set_id):
            raise ValueError(
                "regional candidate input contains multiple candidate sets for "
                f"target/cluster {cluster_key}: {previous!r}, {candidate_set_id!r}"
            )
        fingerprint = str(group["candidate_set_fingerprint"].unique().item())
        if not fingerprint.startswith("sha256:") or len(fingerprint) != 71:
            raise ValueError(
                f"candidate set {candidate_set_id!r} has an invalid fingerprint"
            )


def _accepted_species_taxonomy(taxa: pl.DataFrame) -> dict[str, _Taxon]:
    _require_columns(
        taxa,
        artifact="taxa",
        required={
            "accepted_taxon_key",
            "scientific_name",
            "rank",
            "taxonomic_status",
            "family",
            "genus",
        },
    )
    output: dict[str, _Taxon] = {}
    for row in taxa.iter_rows(named=True):
        if str(row.get("rank") or "").strip().upper() != "SPECIES":
            continue
        if str(row.get("taxonomic_status") or "").strip().upper() != "ACCEPTED":
            continue
        if "in_scope" in taxa.columns and row.get("in_scope") is not True:
            continue
        key = _required_text(row.get("accepted_taxon_key"), field="accepted_taxon_key")
        if key in output:
            raise ValueError(f"duplicate accepted species in taxa: {key}")
        output[key] = _Taxon(
            accepted_taxon_key=key,
            scientific_name=_required_text(row.get("scientific_name"), field="scientific_name"),
            family=_required_text(row.get("family"), field="family"),
            genus=_required_text(row.get("genus"), field="genus"),
        )
    return output


def _validated_clusters(
    clusters: pl.DataFrame,
    *,
    target_key: str,
) -> dict[str, str | None]:
    _require_columns(
        clusters,
        artifact="flickr_geo_clusters",
        required={
            "geo_cluster_id",
            "target_accepted_taxon_key",
            "candidate_distribution_only",
        },
    )
    output: dict[str, str | None] = {}
    for row in clusters.iter_rows(named=True):
        cluster_id = _required_text(row.get("geo_cluster_id"), field="geo_cluster_id")
        if cluster_id in output:
            raise ValueError(f"duplicate Flickr geo cluster ID: {cluster_id}")
        row_target = _required_text(
            row.get("target_accepted_taxon_key"),
            field="target_accepted_taxon_key",
        )
        if row_target != target_key:
            raise ValueError(
                f"Flickr geo cluster {cluster_id!r} target {row_target!r} does not match "
                f"requested target {target_key!r}"
            )
        if row.get("candidate_distribution_only") is not True:
            raise ValueError(
                f"Flickr geo cluster {cluster_id!r} is not candidate-distribution evidence"
            )
        output[cluster_id] = _optional_text(row.get("cluster_configuration_hash"))
    return output


def _regional_geography(
    frame: pl.DataFrame,
    *,
    taxonomy: dict[str, _Taxon],
    registry_version: str,
    cluster_ids: set[str],
) -> dict[str, dict[str, _GeoEvidence]]:
    if frame.is_empty() and not frame.columns:
        return {}
    _require_columns(
        frame,
        artifact="regional_taxon_occurrence",
        required={
            "regional_scope_id",
            "regional_scope_type",
            "accepted_taxon_key",
            "scientific_name",
            "family",
            "genus",
            "occurrence_count",
            "coordinate_confidence",
            "overlap_type",
            "source",
            "evidence_version",
            "registry_version",
        },
    )
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    primary_keys: set[tuple[str, str, str, str]] = set()
    evidence_versions_by_source: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in frame.iter_rows(named=True):
        if (
            "schema_version" in frame.columns
            and row.get("schema_version") != REGIONAL_TAXON_OCCURRENCE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported regional taxon occurrence schema version")
        if row.get("regional_scope_type") != "geo_cluster":
            continue
        cluster_id = _required_text(row.get("regional_scope_id"), field="regional_scope_id")
        if cluster_id not in cluster_ids:
            continue
        key = _required_text(row.get("accepted_taxon_key"), field="accepted_taxon_key")
        taxon = taxonomy.get(key)
        if taxon is None:
            raise ValueError(f"regional occurrence taxon is not accepted in the registry: {key}")
        if _required_text(row.get("registry_version"), field="registry_version") != registry_version:
            raise ValueError(f"regional occurrence registry version conflicts for {key}")
        for field_name, expected in (
            ("scientific_name", taxon.scientific_name),
            ("family", taxon.family),
            ("genus", taxon.genus),
        ):
            if _required_text(row.get(field_name), field=field_name) != expected:
                raise ValueError(f"regional occurrence {key} conflicts with accepted {field_name}")
        overlap = _required_text(row.get("overlap_type"), field="overlap_type")
        if overlap not in _OVERLAP_PRIORITY:
            raise ValueError(f"unsupported regional overlap type: {overlap}")
        source = _required_text(row.get("source"), field="source")
        evidence_version = _required_text(
            row.get("evidence_version"),
            field="evidence_version",
        )
        primary_key = (cluster_id, key, source, evidence_version)
        if primary_key in primary_keys:
            raise ValueError(f"duplicate regional occurrence primary key: {primary_key}")
        primary_keys.add(primary_key)
        evidence_versions_by_source[(cluster_id, key, source)].add(evidence_version)
        grouped[(cluster_id, key)].append(dict(row))

    mixed_versions = sorted(
        key
        for key, versions in evidence_versions_by_source.items()
        if len(versions) > 1
    )
    if mixed_versions:
        raise ValueError(
            "regional occurrence input mixes evidence versions for a source: "
            f"{mixed_versions[0]}"
        )

    output: dict[str, dict[str, _GeoEvidence]] = defaultdict(dict)
    for (cluster_id, key), rows in sorted(grouped.items()):
        strongest = min(_OVERLAP_PRIORITY[str(row["overlap_type"])] for row in rows)
        selected = [
            row
            for row in rows
            if _OVERLAP_PRIORITY[str(row["overlap_type"])] == strongest
        ]
        overlap = str(selected[0]["overlap_type"])
        counts = [
            _positive_int(row.get("occurrence_count"), field="occurrence_count")
            for row in selected
        ]
        support = sum(counts)
        confidences = [
            _optional_score(row.get("coordinate_confidence"), field="coordinate_confidence")
            for row in selected
        ]
        score = None
        if support and all(value is not None for value in confidences):
            weighted_confidence = sum(
                float(confidence) * count
                for confidence, count in zip(confidences, counts, strict=True)
            ) / support
            score = _OVERLAP_WEIGHT[overlap] * weighted_confidence
        output[cluster_id][key] = _GeoEvidence(
            overlap_type=overlap,
            occurrence_support=support,
            geographic_evidence_score=score,
            evidence_versions=tuple(
                sorted({_required_text(row["evidence_version"], field="evidence_version") for row in selected})
            ),
        )
    return {cluster_id: dict(items) for cluster_id, items in output.items()}


def _relationship_candidates(
    frame: pl.DataFrame | None,
    *,
    target_key: str,
    taxonomy: dict[str, _Taxon],
) -> tuple[dict[str, set[str]], set[str]]:
    if frame is None:
        return {}, set()
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("competitor_relationships must be a Polars DataFrame or None")
    if frame.is_empty():
        return {}, set()
    _require_columns(
        frame,
        artifact="competitor_relationships",
        required={
            "subject_accepted_taxon_key",
            "object_scope_type",
            "object_scope_id",
            "relationship_type",
            "evidence_version",
            "review_status",
            "enabled",
        },
    )
    reasons: dict[str, set[str]] = defaultdict(set)
    versions: set[str] = set()
    for row in frame.iter_rows(named=True):
        if row.get("enabled") is not True:
            continue
        subject = _required_text(
            row.get("subject_accepted_taxon_key"),
            field="subject_accepted_taxon_key",
        )
        if subject != target_key:
            continue
        if str(row.get("review_status") or "").strip().casefold() != "reviewed":
            raise ValueError("enabled competitor relationship must be reviewed")
        relationship_type = _required_text(
            row.get("relationship_type"),
            field="relationship_type",
        )
        reason = _RELATIONSHIP_REASON.get(relationship_type)
        if reason is None:
            raise ValueError(f"unsupported competitor relationship type: {relationship_type}")
        scope_type = _required_text(row.get("object_scope_type"), field="object_scope_type")
        scope_id = _required_text(row.get("object_scope_id"), field="object_scope_id")
        if scope_type == "species":
            keys = [scope_id] if scope_id in taxonomy else []
        elif scope_type == "genus":
            keys = sorted(key for key, taxon in taxonomy.items() if taxon.genus == scope_id)
        else:
            raise ValueError(f"unsupported competitor object scope type: {scope_type}")
        if not keys:
            raise ValueError(
                f"competitor relationship object is not accepted in the registry: {scope_type}:{scope_id}"
            )
        for key in keys:
            if key != target_key:
                reasons[key].add(reason)
        versions.add(
            "relationships:"
            + _required_text(row.get("evidence_version"), field="evidence_version")
        )
    return {key: set(values) for key, values in reasons.items()}, versions


def _accepted_candidate_keys(
    values: Sequence[str],
    *,
    taxonomy: dict[str, _Taxon],
    field: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence of taxon keys")
    keys = tuple(sorted({_required_text(value, field=field) for value in values}))
    unknown = sorted(set(keys) - set(taxonomy))
    if unknown:
        raise ValueError(f"{field} contains keys that are not accepted species: {unknown}")
    return keys


def _required_dependency_version(
    keys: Sequence[str],
    value: str | None,
    *,
    field: str,
    prefix: str,
) -> str | None:
    version = _optional_text(value)
    if keys and version is None:
        raise ValueError(f"{field} is required when its candidate keys are nonempty")
    return f"{prefix}:{version}" if version else None


def _candidate_order_key(
    candidate: _Candidate,
    *,
    target: _Taxon,
    geography: _GeoEvidence | None,
) -> tuple[object, ...]:
    reason_priority = min(_REASON_PRIORITY[reason] for reason in candidate.reasons)
    score = geography.geographic_evidence_score if geography is not None else None
    support = geography.occurrence_support if geography is not None else 0
    return (
        reason_priority,
        -(score if score is not None else -1.0),
        -support,
        candidate.taxon.family != target.family,
        candidate.taxon.genus != target.genus,
        candidate.taxon.accepted_taxon_key,
    )


def _candidate_row(
    candidate: _Candidate,
    *,
    target: _Taxon,
    target_key: str,
    cluster_id: str,
    geography: _GeoEvidence | None,
    candidate_priority: int,
    source_versions: list[str],
) -> dict[str, object]:
    key = candidate.taxon.accepted_taxon_key
    return {
        "schema_version": REGIONAL_CANDIDATE_SPECIES_SCHEMA_VERSION,
        "candidate_set_id": None,
        "target_accepted_taxon_key": target_key,
        "geo_cluster_id": cluster_id,
        "candidate_accepted_taxon_key": key,
        "scientific_name": candidate.taxon.scientific_name,
        "family": candidate.taxon.family,
        "genus": candidate.taxon.genus,
        "candidate_reason": sorted(
            candidate.reasons,
            key=lambda reason: (_REASON_PRIORITY[reason], reason),
        ),
        "geographic_evidence_score": (
            _float32(geography.geographic_evidence_score)
            if geography is not None and geography.geographic_evidence_score is not None
            else None
        ),
        "occurrence_support": geography.occurrence_support if geography is not None else 0,
        "same_genus": candidate.taxon.genus == target.genus,
        "same_family": candidate.taxon.family == target.family,
        "known_mimic": candidate.known_mimic,
        "historical_false_positive": candidate.historical_false_positive,
        "visually_nearest": candidate.visually_nearest,
        "target_candidate": key == target_key,
        "candidate_priority": candidate_priority,
        "source_versions": source_versions,
        "candidate_set_fingerprint": None,
    }


def _candidate_set_fingerprint(
    *,
    target_key: str,
    cluster_id: str,
    config: RegionalCandidateConfig,
    source_versions: list[str],
    candidates: list[dict[str, object]],
) -> str:
    payload = {
        "schema_version": REGIONAL_CANDIDATE_SPECIES_SCHEMA_VERSION,
        "policy_version": config.policy_version,
        "target_accepted_taxon_key": target_key,
        "geo_cluster_id": cluster_id,
        "minimum_local_same_family_candidates": config.minimum_local_same_family_candidates,
        "include_registry_same_family_for_no_geo": (
            config.include_registry_same_family_for_no_geo
        ),
        "source_versions": source_versions,
        "candidates": [
            {
                key: value
                for key, value in row.items()
                if key not in {"candidate_set_id", "candidate_set_fingerprint"}
            }
            for row in candidates
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _candidate_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = regional_candidate_species_schema()
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).sort(
        ["candidate_set_id", "candidate_priority", "candidate_accepted_taxon_key"]
    )


def _require_columns(
    frame: pl.DataFrame,
    *,
    artifact: str,
    required: set[str],
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{artifact} is missing required columns: {missing}")


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _optional_score(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric or null") from exc
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return score


def _versions(values: Sequence[str], *, field: str) -> set[str]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence")
    return {_required_text(value, field=field) for value in values}


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


__all__ = [
    "GEOGRAPHIC_EVIDENCE_SCORE_VERSION",
    "REGIONAL_CANDIDATE_POLICY_VERSION",
    "REGIONAL_CANDIDATE_SPECIES_FILE",
    "REGIONAL_CANDIDATE_SPECIES_SCHEMA_VERSION",
    "RegionalCandidateConfig",
    "build_regional_candidate_species",
    "regional_candidate_species_schema",
    "validate_regional_candidate_species",
    "write_regional_candidate_species",
]
