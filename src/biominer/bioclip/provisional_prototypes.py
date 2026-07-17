"""Robust prototypes for provider-asserted and mixed reference banks."""

from __future__ import annotations

from array import array
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from math import fsum, isfinite, sqrt

import polars as pl

from biominer.bioclip.reference_embeddings import (
    reference_embeddings_artifact_fingerprint,
    validate_reference_embeddings,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint


PROVISIONAL_PROTOTYPES_SCHEMA_VERSION = "provisional-prototypes-v1.0.0"
PROVISIONAL_PROTOTYPES_FILE = "provisional_reference_prototypes.parquet"
ROBUST_PROTOTYPE_METHODS = (
    "normalized_mean",
    "mean_centered_mean",
    "trimmed_mean",
    "medoid",
)


@dataclass(frozen=True, slots=True)
class RobustPrototypePolicy:
    """Deterministic robust aggregation and within-class clustering policy."""

    methods: tuple[str, ...] = ROBUST_PROTOTYPE_METHODS
    trim_fraction: float = 0.10
    maximum_observations_per_species_route: int = 64
    prototype_count: int = 1
    minimum_cluster_size: int = 2
    clustering_iterations: int = 20
    seed: int = 42

    def __post_init__(self) -> None:
        methods = tuple(dict.fromkeys(self.methods))
        if not methods or set(methods) - set(ROBUST_PROTOTYPE_METHODS):
            raise ValueError("unsupported robust prototype method")
        if not 0 <= self.trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        for field in (
            "maximum_observations_per_species_route",
            "prototype_count",
            "minimum_cluster_size",
            "clustering_iterations",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        object.__setattr__(self, "methods", methods)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": "robust-prototype-policy-v1",
                "methods": list(self.methods),
                "trim_fraction": self.trim_fraction,
                "maximum_observations_per_species_route": (
                    self.maximum_observations_per_species_route
                ),
                "prototype_count": self.prototype_count,
                "minimum_cluster_size": self.minimum_cluster_size,
                "clustering_iterations": self.clustering_iterations,
                "seed": self.seed,
            }
        )


def provisional_prototypes_schema(dimension: int) -> dict[str, pl.DataType]:
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("prototype embedding dimension must be positive")
    return {
        "schema_version": pl.String,
        "prototype_id": pl.String,
        "accepted_taxon_key": pl.String,
        "species": pl.String,
        "route": pl.String,
        "visual_input_kind": pl.String,
        "prototype_method": pl.String,
        "prototype_index": pl.UInt16,
        "prototype_count": pl.UInt16,
        "member_reference_media_ids": pl.List(pl.String),
        "member_observation_ids": pl.List(pl.String),
        "reference_count": pl.UInt32,
        "independent_observation_count": pl.UInt32,
        "provisional_reference_count": pl.UInt32,
        "human_verified_reference_count": pl.UInt32,
        "outlier_count": pl.UInt32,
        "dispersion": pl.Float64,
        "balanced_sampling_seed": pl.UInt64,
        "trim_fraction": pl.Float64,
        "policy_fingerprint": pl.String,
        "reference_admission_mode": pl.String,
        "admission_policy_fingerprint": pl.String,
        "embedding_dimension": pl.UInt32,
        "embedding": pl.Array(pl.Float32, dimension),
        "embedding_norm": pl.Float64,
        "model_fingerprint": pl.String,
        "reference_embedding_fingerprint": pl.String,
        "support_manifest_fingerprint": pl.String,
        "prototype_fingerprint": pl.String,
    }


def build_robust_provisional_prototypes(
    reference_embeddings: pl.DataFrame,
    *,
    policy: RobustPrototypePolicy | None = None,
) -> pl.DataFrame:
    """Build robust, evidence-counted species/route prototypes."""

    validate_reference_embeddings(reference_embeddings)
    active = policy or RobustPrototypePolicy()
    support = reference_embeddings.filter(
        pl.col("support_split") == "support_train"
    )
    if support.is_empty():
        raise ValueError("robust prototypes require support_train embeddings")
    dimension = int(support["embedding_dimension"][0])
    observations = _observation_rows(support, dimension=dimension)
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in observations:
        grouped[
            (
                str(row["accepted_taxon_key"]),
                str(row["species"]),
                str(row["route"]),
                str(row["visual_input_kind"]),
            )
        ].append(row)
    selected = {
        key: _subsample(values, policy=active)
        for key, values in sorted(grouped.items())
    }
    centers = _balanced_centers(selected, dimension=dimension, policy=active)
    model_fingerprint = _single(support, "model_fingerprint")
    support_fingerprint = _single(support, "support_manifest_fingerprint")
    admission_mode = _single(support, "reference_admission_mode")
    admission_fingerprint = _single(support, "admission_policy_fingerprint")
    embedding_fingerprint = reference_embeddings_artifact_fingerprint(support)
    output: list[dict[str, object]] = []
    for key, members in selected.items():
        clusters = _clusters(members, policy=active)
        for cluster_index, cluster in enumerate(clusters):
            for method in active.methods:
                vector = _prototype_vector(
                    cluster,
                    method=method,
                    trim_fraction=active.trim_fraction,
                    center=centers[(key[2], key[3])],
                )
                stored, norm = _unit_float32(vector)
                row: dict[str, object] = {
                    "schema_version": PROVISIONAL_PROTOTYPES_SCHEMA_VERSION,
                    "prototype_id": "",
                    "accepted_taxon_key": key[0],
                    "species": key[1],
                    "route": key[2],
                    "visual_input_kind": key[3],
                    "prototype_method": method,
                    "prototype_index": cluster_index,
                    "prototype_count": len(clusters),
                    "member_reference_media_ids": sorted(
                        {
                            media_id
                            for item in cluster
                            for media_id in item["reference_media_ids"]
                        }
                    ),
                    "member_observation_ids": sorted(
                        str(item["reference_observation_id"]) for item in cluster
                    ),
                    "reference_count": sum(
                        len(item["reference_media_ids"]) for item in cluster
                    ),
                    "independent_observation_count": len(cluster),
                    "provisional_reference_count": len(
                        {
                            media_id
                            for item in cluster
                            for media_id in item["provisional_media_ids"]
                        }
                    ),
                    "human_verified_reference_count": len(
                        {
                            media_id
                            for item in cluster
                            for media_id in item["human_verified_media_ids"]
                        }
                    ),
                    "outlier_count": len(
                        {
                            media_id
                            for item in cluster
                            for media_id in item["outlier_media_ids"]
                        }
                    ),
                    "dispersion": _dispersion(cluster, stored),
                    "balanced_sampling_seed": active.seed,
                    "trim_fraction": active.trim_fraction,
                    "policy_fingerprint": active.fingerprint,
                    "reference_admission_mode": admission_mode,
                    "admission_policy_fingerprint": admission_fingerprint,
                    "embedding_dimension": dimension,
                    "embedding": list(stored),
                    "embedding_norm": norm,
                    "model_fingerprint": model_fingerprint,
                    "reference_embedding_fingerprint": embedding_fingerprint,
                    "support_manifest_fingerprint": support_fingerprint,
                    "prototype_fingerprint": "",
                }
                identity = {key: value for key, value in row.items() if key not in {"prototype_id", "prototype_fingerprint"}}
                row["prototype_id"] = canonical_semantic_fingerprint(identity)
                fingerprint_payload = dict(row)
                fingerprint_payload.pop("prototype_fingerprint")
                row["prototype_fingerprint"] = canonical_semantic_fingerprint(
                    fingerprint_payload
                )
                output.append(row)
    frame = pl.DataFrame(
        output,
        schema=provisional_prototypes_schema(dimension),
        orient="row",
        strict=True,
    ).sort(
        "accepted_taxon_key",
        "route",
        "visual_input_kind",
        "prototype_index",
        "prototype_method",
    )
    validate_robust_provisional_prototypes(frame)
    return frame


def validate_robust_provisional_prototypes(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame) or frame.is_empty():
        raise ValueError("provisional prototypes must be a non-empty Polars frame")
    dimensions = frame["embedding_dimension"].unique().to_list()
    if len(dimensions) != 1 or frame.schema != provisional_prototypes_schema(
        int(dimensions[0])
    ):
        raise ValueError("provisional prototype schema mismatch")
    if frame["prototype_id"].n_unique() != frame.height:
        raise ValueError("provisional prototype IDs must be unique")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != PROVISIONAL_PROTOTYPES_SCHEMA_VERSION:
            raise ValueError("unsupported provisional prototype schema")
        if row["prototype_method"] not in ROBUST_PROTOTYPE_METHODS:
            raise ValueError("unsupported provisional prototype method")
        if int(row["provisional_reference_count"]) + int(
            row["human_verified_reference_count"]
        ) != int(row["reference_count"]):
            raise ValueError("prototype evidence counts are inconsistent")
        if int(row["outlier_count"]) > int(row["reference_count"]):
            raise ValueError("prototype outlier count exceeds support")
        if not isfinite(float(row["dispersion"])) or not 0 <= float(
            row["dispersion"]
        ) <= 2:
            raise ValueError("prototype dispersion is invalid")
        vector = tuple(float(value) for value in row["embedding"])
        norm = sqrt(fsum(value * value for value in vector))
        if abs(norm - 1.0) > 1e-5 or abs(norm - float(row["embedding_norm"])) > 1e-12:
            raise ValueError("provisional prototype is not unit normalized")
        payload = dict(row)
        fingerprint = payload.pop("prototype_fingerprint")
        if fingerprint != canonical_semantic_fingerprint(payload):
            raise ValueError("provisional prototype fingerprint mismatch")


def _observation_rows(
    frame: pl.DataFrame,
    *,
    dimension: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in frame.iter_rows(named=True):
        key = (
            str(row["accepted_taxon_key"]),
            str(row["scientific_name"]),
            str(row["route"]),
            str(row["visual_input_kind"]),
            str(row["reference_observation_id"]),
        )
        grouped[key].append(row)
    result: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        media_ids = {str(row["reference_media_id"]) for row in rows}
        provisional = {
            str(row["reference_media_id"])
            for row in rows
            if row["provisional_support"]
        }
        human = media_ids - provisional
        outliers = {
            str(row["reference_media_id"])
            for row in rows
            if set(row["reference_quality_flags"])
            & {"embedding_outlier", "reference_outlier"}
        }
        result.append(
            {
                "accepted_taxon_key": key[0],
                "species": key[1],
                "route": key[2],
                "visual_input_kind": key[3],
                "reference_observation_id": key[4],
                "reference_media_ids": tuple(sorted(media_ids)),
                "provisional_media_ids": tuple(sorted(provisional)),
                "human_verified_media_ids": tuple(sorted(human)),
                "outlier_media_ids": tuple(sorted(outliers)),
                "embedding": _unit(
                    [
                        fsum(float(row["embedding"][index]) for row in rows)
                        / len(rows)
                        for index in range(dimension)
                    ]
                ),
                "rank_fingerprint": min(
                    str(row["embedding_fingerprint"]) for row in rows
                ),
            }
        )
    return result


def _subsample(
    rows: Sequence[dict[str, object]],
    *,
    policy: RobustPrototypePolicy,
) -> list[dict[str, object]]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{policy.seed}:{row['reference_observation_id']}:{row['rank_fingerprint']}".encode()
        ).hexdigest(),
    )
    return ranked[: policy.maximum_observations_per_species_route]


def _balanced_centers(
    groups: Mapping[tuple[str, str, str, str], Sequence[dict[str, object]]],
    *,
    dimension: int,
    policy: RobustPrototypePolicy,
) -> dict[tuple[str, str], tuple[float, ...]]:
    by_contract: dict[tuple[str, str], list[Sequence[dict[str, object]]]] = defaultdict(list)
    for key, rows in groups.items():
        by_contract[(key[2], key[3])].append(rows)
    centers: dict[tuple[str, str], tuple[float, ...]] = {}
    for contract, species_groups in by_contract.items():
        balanced = min(len(rows) for rows in species_groups)
        selected = [row for rows in species_groups for row in rows[:balanced]]
        centers[contract] = tuple(
            fsum(float(row["embedding"][index]) for row in selected)
            / len(selected)
            for index in range(dimension)
        )
    return centers


def _clusters(
    rows: Sequence[dict[str, object]],
    *,
    policy: RobustPrototypePolicy,
) -> list[list[dict[str, object]]]:
    maximum = min(policy.prototype_count, len(rows) // policy.minimum_cluster_size)
    if maximum < 2:
        return [list(rows)]
    centers = [rows[0]]
    while len(centers) < maximum:
        centers.append(
            max(
                (row for row in rows if row not in centers),
                key=lambda row: (
                    min(
                        1 - _dot(row["embedding"], center["embedding"])
                        for center in centers
                    ),
                    str(row["reference_observation_id"]),
                ),
            )
        )
    assignments: list[int] = []
    for _ in range(policy.clustering_iterations):
        assignments = [
            max(
                range(len(centers)),
                key=lambda index: (
                    _dot(row["embedding"], centers[index]["embedding"]),
                    -index,
                ),
            )
            for row in rows
        ]
        groups = [
            [row for row, assigned in zip(rows, assignments, strict=True) if assigned == index]
            for index in range(len(centers))
        ]
        if any(len(group) < policy.minimum_cluster_size for group in groups):
            return [list(rows)]
        updated = [
            {**group[0], "embedding": _unit(_coordinate_mean(group))}
            for group in groups
        ]
        if all(
            _dot(old["embedding"], new["embedding"]) > 1 - 1e-10
            for old, new in zip(centers, updated, strict=True)
        ):
            break
        centers = updated
    result = [
        [row for row, assigned in zip(rows, assignments, strict=True) if assigned == index]
        for index in range(len(centers))
    ]
    return sorted(result, key=lambda group: str(group[0]["reference_observation_id"]))


def _prototype_vector(
    rows: Sequence[dict[str, object]],
    *,
    method: str,
    trim_fraction: float,
    center: Sequence[float],
) -> tuple[float, ...]:
    vectors = [tuple(float(value) for value in row["embedding"]) for row in rows]
    if method == "medoid":
        return min(
            vectors,
            key=lambda vector: (
                fsum(1 - _dot(vector, other) for other in vectors),
                vector,
            ),
        )
    if method == "trimmed_mean":
        cut = min(int(len(vectors) * trim_fraction), (len(vectors) - 1) // 2)
        values = [
            sorted(vector[index] for vector in vectors)[cut : len(vectors) - cut]
            for index in range(len(vectors[0]))
        ]
        return tuple(fsum(items) / len(items) for items in values)
    mean = tuple(
        fsum(vector[index] for vector in vectors) / len(vectors)
        for index in range(len(vectors[0]))
    )
    if method == "mean_centered_mean":
        centered = tuple(value - center[index] for index, value in enumerate(mean))
        if sqrt(fsum(value * value for value in centered)) > 1e-12:
            return centered
    return mean


def _coordinate_mean(rows: Sequence[dict[str, object]]) -> list[float]:
    dimension = len(rows[0]["embedding"])
    return [
        fsum(float(row["embedding"][index]) for row in rows) / len(rows)
        for index in range(dimension)
    ]


def _dispersion(
    rows: Sequence[dict[str, object]],
    prototype: Sequence[float],
) -> float:
    return fsum(1 - _dot(row["embedding"], prototype) for row in rows) / len(rows)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return fsum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _unit(values: Sequence[float]) -> tuple[float, ...]:
    norm = sqrt(fsum(float(value) ** 2 for value in values))
    if not isfinite(norm) or norm <= 1e-12:
        raise ValueError("prototype vector must have non-zero finite norm")
    return tuple(float(value) / norm for value in values)


def _unit_float32(values: Sequence[float]) -> tuple[tuple[float, ...], float]:
    normalized = _unit(values)
    stored = tuple(float(value) for value in array("f", normalized))
    norm = sqrt(fsum(value * value for value in stored))
    return stored, norm


def _single(frame: pl.DataFrame, field: str) -> str:
    values = frame[field].unique().to_list()
    if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
        raise ValueError(f"{field} must have one nonblank value")
    return values[0]


__all__ = [
    "PROVISIONAL_PROTOTYPES_FILE",
    "PROVISIONAL_PROTOTYPES_SCHEMA_VERSION",
    "ROBUST_PROTOTYPE_METHODS",
    "RobustPrototypePolicy",
    "build_robust_provisional_prototypes",
    "provisional_prototypes_schema",
    "validate_robust_provisional_prototypes",
]
