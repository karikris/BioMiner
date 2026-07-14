"""Versioned visual-neighbour species graph from frozen global prototypes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
from math import fsum, isfinite
from pathlib import Path
import re
import struct

import polars as pl

from biominer.bioclip.reference_prototypes import (
    PROTOTYPE_AGGREGATE_VIEW,
    PROTOTYPE_GLOBAL_GEO_CLUSTER_ID,
    PROTOTYPE_KIND_AGGREGATE,
    PROTOTYPE_METHOD_NORMALIZED_MEAN,
    PROTOTYPE_SCOPE_GLOBAL,
    REFERENCE_PROTOTYPE_METHODS,
    load_reference_prototypes,
    reference_prototypes_artifact_fingerprint,
    validate_reference_prototypes,
)
from biominer.common.semantic_hash import (
    canonical_semantic_bytes,
    canonical_semantic_fingerprint,
)
from biominer.references.readiness import REFERENCE_ROUTES
from biominer.storage.parquet import write_parquet


VISUAL_NEIGHBOUR_SPECIES_SCHEMA_VERSION = "visual-neighbour-species-v1.0.0"
VISUAL_NEIGHBOUR_GRAPH_ALGORITHM_VERSION = "global-aggregate-cosine-knn-v1"
VISUAL_NEIGHBOUR_GRAPH_CONFIG_SCHEMA_VERSION = "visual-neighbour-graph-config-v1"
VISUAL_NEIGHBOUR_GRAPH_FINGERPRINT_SCHEMA_VERSION = (
    "visual-neighbour-graph-fingerprint-v1"
)
VISUAL_NEIGHBOUR_EDGE_ID_SCHEMA_VERSION = "visual-neighbour-edge-id-v1"
VISUAL_NEIGHBOUR_SPECIES_FILE = "visual_neighbour_species.parquet"

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VisualNeighbourGraphConfig:
    """Species-level neighbour selection over one aggregate prototype per class."""

    top_k_neighbors: int = 10
    minimum_similarity: float = -1.0
    prototype_method: str = PROTOTYPE_METHOD_NORMALIZED_MEAN

    def __post_init__(self) -> None:
        top_k = _positive_integer(self.top_k_neighbors, field="top_k_neighbors")
        similarity = _finite_float(
            self.minimum_similarity,
            field="minimum_similarity",
        )
        if not -1.0 <= similarity <= 1.0:
            raise ValueError("minimum_similarity must be in [-1, 1]")
        if self.prototype_method not in REFERENCE_PROTOTYPE_METHODS:
            raise ValueError(
                f"unsupported visual-neighbour prototype method: {self.prototype_method}"
            )
        object.__setattr__(self, "top_k_neighbors", top_k)
        object.__setattr__(self, "minimum_similarity", similarity)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": VISUAL_NEIGHBOUR_GRAPH_CONFIG_SCHEMA_VERSION,
                "algorithm_version": VISUAL_NEIGHBOUR_GRAPH_ALGORITHM_VERSION,
                "top_k_neighbors": self.top_k_neighbors,
                "minimum_similarity": self.minimum_similarity,
                "prototype_kind": PROTOTYPE_KIND_AGGREGATE,
                "prototype_method": self.prototype_method,
                "cluster_scope_type": PROTOTYPE_SCOPE_GLOBAL,
                "geo_cluster_id": PROTOTYPE_GLOBAL_GEO_CLUSTER_ID,
                "view": PROTOTYPE_AGGREGATE_VIEW,
            }
        )


def visual_neighbour_species_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "graph_version": pl.String,
        "algorithm_version": pl.String,
        "graph_configuration_fingerprint": pl.String,
        "graph_fingerprint": pl.String,
        "configured_top_k": pl.UInt32,
        "configured_minimum_similarity": pl.Float64,
        "edge_id": pl.String,
        "subject_accepted_taxon_key": pl.String,
        "subject_scientific_name": pl.String,
        "neighbour_accepted_taxon_key": pl.String,
        "neighbour_scientific_name": pl.String,
        "route": pl.String,
        "visual_input_kind": pl.String,
        "prototype_kind": pl.String,
        "prototype_method": pl.String,
        "subject_prototype_id": pl.String,
        "neighbour_prototype_id": pl.String,
        "best_prototype_similarity": pl.Float64,
        "neighbour_rank": pl.UInt32,
        "supporting_prototype_pair_count": pl.UInt32,
        "supporting_prototype_pairs": pl.List(
            pl.Struct(
                {
                    "subject_prototype_id": pl.String,
                    "neighbour_prototype_id": pl.String,
                    "similarity": pl.Float64,
                }
            )
        ),
        "embedding_dimension": pl.UInt32,
        "model_fingerprint": pl.String,
        "reference_embedding_fingerprint": pl.String,
        "reference_prototypes_fingerprint": pl.String,
        "support_manifest_fingerprint": pl.String,
        "edge_fingerprint": pl.String,
    }


def build_visual_neighbour_species(
    reference_prototypes: pl.DataFrame | str | Path,
    *,
    graph_version: str,
    config: VisualNeighbourGraphConfig | None = None,
) -> pl.DataFrame:
    """Build directed species edges within each route and visual-input contract."""

    version = _required_text(graph_version, field="graph_version")
    effective_config = config or VisualNeighbourGraphConfig()
    if not isinstance(effective_config, VisualNeighbourGraphConfig):
        raise TypeError("config must be a VisualNeighbourGraphConfig")
    prototypes = _prototype_frame(reference_prototypes)
    prototypes_fingerprint = reference_prototypes_artifact_fingerprint(prototypes)
    selected = prototypes.filter(
        (pl.col("cluster_scope_type") == PROTOTYPE_SCOPE_GLOBAL)
        & (pl.col("geo_cluster_id") == PROTOTYPE_GLOBAL_GEO_CLUSTER_ID)
        & (pl.col("view") == PROTOTYPE_AGGREGATE_VIEW)
        & (pl.col("prototype_kind") == PROTOTYPE_KIND_AGGREGATE)
        & (pl.col("prototype_method") == effective_config.prototype_method)
    )
    if selected.is_empty():
        raise ValueError(
            "visual-neighbour graph has no matching global aggregate prototypes"
        )
    model_fingerprint = _single_sha256(selected, "model_fingerprint")
    embedding_fingerprint = _single_sha256(
        selected,
        "reference_embedding_fingerprint",
    )
    support_manifest_fingerprint = _single_sha256(
        selected,
        "support_manifest_fingerprint",
    )
    dimensions = selected["embedding_dimension"].unique().to_list()
    if len(dimensions) != 1:
        raise ValueError("visual-neighbour prototypes have mixed dimensions")
    embedding_dimension = _positive_integer(
        dimensions[0],
        field="embedding_dimension",
    )

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in selected.iter_rows(named=True):
        identity = (
            str(row["route"]),
            str(row["visual_input_kind"]),
            str(row["accepted_taxon_key"]),
        )
        if identity in seen:
            raise ValueError(
                "visual-neighbour graph requires one aggregate prototype per species contract"
            )
        seen.add(identity)
        grouped[identity[:2]].append(row)

    rows: list[dict[str, object]] = []
    for _, items in sorted(grouped.items()):
        ordered = sorted(
            items,
            key=lambda row: (
                str(row["accepted_taxon_key"]),
                str(row["prototype_id"]),
            ),
        )
        if len(ordered) < 2:
            continue
        for subject in ordered:
            candidates: list[tuple[float, str, str, dict[str, object]]] = []
            subject_key = str(subject["accepted_taxon_key"])
            for neighbour in ordered:
                neighbour_key = str(neighbour["accepted_taxon_key"])
                if neighbour_key == subject_key:
                    continue
                similarity = _cosine_similarity(
                    subject["embedding"],
                    neighbour["embedding"],
                )
                if similarity < effective_config.minimum_similarity:
                    continue
                candidates.append(
                    (
                        similarity,
                        neighbour_key,
                        str(neighbour["prototype_id"]),
                        neighbour,
                    )
                )
            candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
            for rank, (similarity, _, _, neighbour) in enumerate(
                candidates[: effective_config.top_k_neighbors],
                start=1,
            ):
                rows.append(
                    _edge_row(
                        graph_version=version,
                        config=effective_config,
                        subject=subject,
                        neighbour=neighbour,
                        similarity=similarity,
                        rank=rank,
                        embedding_dimension=embedding_dimension,
                        model_fingerprint=model_fingerprint,
                        reference_embedding_fingerprint=embedding_fingerprint,
                        reference_prototypes_fingerprint=prototypes_fingerprint,
                        support_manifest_fingerprint=support_manifest_fingerprint,
                    )
                )
    if not rows:
        raise ValueError(
            "visual-neighbour graph requires at least two species in one scoring contract"
        )
    graph_fingerprint = _graph_fingerprint(
        graph_version=version,
        configuration_fingerprint=effective_config.fingerprint,
        reference_prototypes_fingerprint=prototypes_fingerprint,
        edge_fingerprints=sorted(str(row["edge_fingerprint"]) for row in rows),
    )
    for row in rows:
        row["graph_fingerprint"] = graph_fingerprint
    result = _sort_graph_frame(
        pl.DataFrame(
            rows,
            schema=visual_neighbour_species_schema(),
            orient="row",
            strict=True,
        )
    )
    validate_visual_neighbour_species(result)
    _log_event(
        "visual_neighbour_graph_built",
        graph_version=version,
        input_prototype_count=prototypes.height,
        selected_aggregate_prototype_count=selected.height,
        output_edge_count=result.height,
        route_count=result["route"].n_unique(),
        graph_fingerprint=graph_fingerprint,
        reference_prototypes_fingerprint=prototypes_fingerprint,
    )
    return result


def validate_visual_neighbour_species(
    frame: pl.DataFrame,
    *,
    expected_reference_prototypes_fingerprint: str | None = None,
) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("visual-neighbour species graph must be a Polars DataFrame")
    if dict(frame.schema) != visual_neighbour_species_schema():
        raise ValueError("visual-neighbour species graph physical schema mismatch")
    if frame.is_empty():
        raise ValueError("visual-neighbour species graph must not be empty")
    if not frame.equals(_sort_graph_frame(frame)):
        raise ValueError(
            "visual-neighbour species graph is not deterministically sorted"
        )
    if frame["edge_id"].n_unique() != frame.height:
        raise ValueError("visual-neighbour species graph contains duplicate edge IDs")
    if frame["edge_fingerprint"].n_unique() != frame.height:
        raise ValueError(
            "visual-neighbour species graph contains duplicate edge fingerprints"
        )

    graph_version = _single_text(frame, "graph_version")
    if _single_text(frame, "algorithm_version") != (
        VISUAL_NEIGHBOUR_GRAPH_ALGORITHM_VERSION
    ):
        raise ValueError("unsupported visual-neighbour graph algorithm")
    top_k = _single_positive_integer(frame, "configured_top_k")
    minimum_similarity = _single_finite_float(
        frame,
        "configured_minimum_similarity",
    )
    prototype_method = _single_text(frame, "prototype_method")
    config = VisualNeighbourGraphConfig(
        top_k_neighbors=top_k,
        minimum_similarity=minimum_similarity,
        prototype_method=prototype_method,
    )
    configuration_fingerprint = _single_sha256(
        frame,
        "graph_configuration_fingerprint",
    )
    if configuration_fingerprint != config.fingerprint:
        raise ValueError("visual-neighbour graph configuration fingerprint mismatch")
    graph_fingerprint = _single_sha256(frame, "graph_fingerprint")
    prototypes_fingerprint = _single_sha256(
        frame,
        "reference_prototypes_fingerprint",
        expected_reference_prototypes_fingerprint,
    )
    _single_sha256(frame, "model_fingerprint")
    _single_sha256(frame, "reference_embedding_fingerprint")
    _single_sha256(frame, "support_manifest_fingerprint")
    dimensions = frame["embedding_dimension"].unique().to_list()
    if len(dimensions) != 1:
        raise ValueError("visual-neighbour graph has mixed embedding dimensions")
    _positive_integer(dimensions[0], field="embedding_dimension")

    ranks: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    neighbours: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != VISUAL_NEIGHBOUR_SPECIES_SCHEMA_VERSION:
            raise ValueError("unsupported visual-neighbour species schema version")
        for field in (
            "subject_accepted_taxon_key",
            "subject_scientific_name",
            "neighbour_accepted_taxon_key",
            "neighbour_scientific_name",
            "route",
            "visual_input_kind",
            "prototype_kind",
            "prototype_method",
            "subject_prototype_id",
            "neighbour_prototype_id",
        ):
            _required_text(row[field], field=field)
        if row["route"] not in REFERENCE_ROUTES:
            raise ValueError("visual-neighbour edge route is invalid")
        if row["prototype_kind"] != PROTOTYPE_KIND_AGGREGATE:
            raise ValueError("visual-neighbour edges must use aggregate prototypes")
        subject_key = str(row["subject_accepted_taxon_key"])
        neighbour_key = str(row["neighbour_accepted_taxon_key"])
        if subject_key == neighbour_key:
            raise ValueError("visual-neighbour graph cannot contain same-species edges")
        similarity = _bounded_similarity(
            row["best_prototype_similarity"],
            field="best_prototype_similarity",
        )
        if similarity < minimum_similarity:
            raise ValueError("visual-neighbour edge is below the configured threshold")
        rank = _positive_integer(row["neighbour_rank"], field="neighbour_rank")
        if rank > top_k:
            raise ValueError("visual-neighbour edge rank exceeds configured top-k")
        pair_count = _positive_integer(
            row["supporting_prototype_pair_count"],
            field="supporting_prototype_pair_count",
        )
        pairs = row["supporting_prototype_pairs"]
        if not isinstance(pairs, list) or len(pairs) != pair_count or pair_count != 1:
            raise ValueError(
                "aggregate visual-neighbour edge requires exactly one prototype pair"
            )
        pair = pairs[0]
        if not isinstance(pair, dict):
            raise ValueError("visual-neighbour prototype pair must be a struct")
        if (
            pair.get("subject_prototype_id") != row["subject_prototype_id"]
            or pair.get("neighbour_prototype_id") != row["neighbour_prototype_id"]
            or _bounded_similarity(pair.get("similarity"), field="pair similarity")
            != similarity
        ):
            raise ValueError("visual-neighbour best prototype pair is inconsistent")
        expected_edge_id = _edge_id(
            graph_version=graph_version,
            configuration_fingerprint=configuration_fingerprint,
            subject_accepted_taxon_key=subject_key,
            neighbour_accepted_taxon_key=neighbour_key,
            route=str(row["route"]),
            visual_input_kind=str(row["visual_input_kind"]),
        )
        if row["edge_id"] != expected_edge_id:
            raise ValueError("visual-neighbour edge ID is invalid")
        if row["edge_fingerprint"] != _edge_fingerprint(row):
            raise ValueError("visual-neighbour edge fingerprint is invalid")
        contract = (str(row["route"]), str(row["visual_input_kind"]), subject_key)
        if neighbour_key in neighbours[contract]:
            raise ValueError("visual-neighbour graph repeats a directed species edge")
        neighbours[contract].add(neighbour_key)
        ranks[contract].append(rank)
    for contract_ranks in ranks.values():
        if sorted(contract_ranks) != list(range(1, len(contract_ranks) + 1)):
            raise ValueError("visual-neighbour ranks must be consecutive per subject")
    expected_graph_fingerprint = _graph_fingerprint(
        graph_version=graph_version,
        configuration_fingerprint=configuration_fingerprint,
        reference_prototypes_fingerprint=prototypes_fingerprint,
        edge_fingerprints=sorted(str(value) for value in frame["edge_fingerprint"]),
    )
    if graph_fingerprint != expected_graph_fingerprint:
        raise ValueError("visual-neighbour graph fingerprint is invalid")


def write_visual_neighbour_species(
    frame: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    validate_visual_neighbour_species(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= VISUAL_NEIGHBOUR_SPECIES_FILE
    written = write_parquet(frame, destination, overwrite=overwrite)
    loaded = pl.read_parquet(written)
    validate_visual_neighbour_species(loaded)
    if not frame.equals(loaded):
        raise ValueError("visual-neighbour graph Parquet round-trip mismatch")
    _log_event(
        "visual_neighbour_graph_written",
        artifact_path=str(written),
        row_count=frame.height,
        byte_count=written.stat().st_size,
        graph_fingerprint=str(frame["graph_fingerprint"][0]),
    )
    return written


def load_visual_neighbour_species(
    path: str | Path,
    *,
    expected_reference_prototypes_fingerprint: str | None = None,
) -> pl.DataFrame:
    source = Path(path)
    if source.is_dir():
        source /= VISUAL_NEIGHBOUR_SPECIES_FILE
    frame = pl.read_parquet(source)
    validate_visual_neighbour_species(
        frame,
        expected_reference_prototypes_fingerprint=(
            expected_reference_prototypes_fingerprint
        ),
    )
    return frame


def _prototype_frame(source: pl.DataFrame | str | Path) -> pl.DataFrame:
    if isinstance(source, pl.DataFrame):
        validate_reference_prototypes(source)
        return source
    return load_reference_prototypes(source)


def _edge_row(
    *,
    graph_version: str,
    config: VisualNeighbourGraphConfig,
    subject: Mapping[str, object],
    neighbour: Mapping[str, object],
    similarity: float,
    rank: int,
    embedding_dimension: int,
    model_fingerprint: str,
    reference_embedding_fingerprint: str,
    reference_prototypes_fingerprint: str,
    support_manifest_fingerprint: str,
) -> dict[str, object]:
    subject_key = str(subject["accepted_taxon_key"])
    neighbour_key = str(neighbour["accepted_taxon_key"])
    route = str(subject["route"])
    visual_input_kind = str(subject["visual_input_kind"])
    subject_prototype_id = str(subject["prototype_id"])
    neighbour_prototype_id = str(neighbour["prototype_id"])
    row: dict[str, object] = {
        "schema_version": VISUAL_NEIGHBOUR_SPECIES_SCHEMA_VERSION,
        "graph_version": graph_version,
        "algorithm_version": VISUAL_NEIGHBOUR_GRAPH_ALGORITHM_VERSION,
        "graph_configuration_fingerprint": config.fingerprint,
        "graph_fingerprint": "",
        "configured_top_k": config.top_k_neighbors,
        "configured_minimum_similarity": config.minimum_similarity,
        "edge_id": _edge_id(
            graph_version=graph_version,
            configuration_fingerprint=config.fingerprint,
            subject_accepted_taxon_key=subject_key,
            neighbour_accepted_taxon_key=neighbour_key,
            route=route,
            visual_input_kind=visual_input_kind,
        ),
        "subject_accepted_taxon_key": subject_key,
        "subject_scientific_name": subject["species"],
        "neighbour_accepted_taxon_key": neighbour_key,
        "neighbour_scientific_name": neighbour["species"],
        "route": route,
        "visual_input_kind": visual_input_kind,
        "prototype_kind": PROTOTYPE_KIND_AGGREGATE,
        "prototype_method": config.prototype_method,
        "subject_prototype_id": subject_prototype_id,
        "neighbour_prototype_id": neighbour_prototype_id,
        "best_prototype_similarity": similarity,
        "neighbour_rank": rank,
        "supporting_prototype_pair_count": 1,
        "supporting_prototype_pairs": [
            {
                "subject_prototype_id": subject_prototype_id,
                "neighbour_prototype_id": neighbour_prototype_id,
                "similarity": similarity,
            }
        ],
        "embedding_dimension": embedding_dimension,
        "model_fingerprint": model_fingerprint,
        "reference_embedding_fingerprint": reference_embedding_fingerprint,
        "reference_prototypes_fingerprint": reference_prototypes_fingerprint,
        "support_manifest_fingerprint": support_manifest_fingerprint,
        "edge_fingerprint": "",
    }
    row["edge_fingerprint"] = _edge_fingerprint(row)
    return row


def _edge_id(
    *,
    graph_version: str,
    configuration_fingerprint: str,
    subject_accepted_taxon_key: str,
    neighbour_accepted_taxon_key: str,
    route: str,
    visual_input_kind: str,
) -> str:
    digest = canonical_semantic_fingerprint(
        {
            "schema_version": VISUAL_NEIGHBOUR_EDGE_ID_SCHEMA_VERSION,
            "graph_version": graph_version,
            "configuration_fingerprint": configuration_fingerprint,
            "subject_accepted_taxon_key": subject_accepted_taxon_key,
            "neighbour_accepted_taxon_key": neighbour_accepted_taxon_key,
            "route": route,
            "visual_input_kind": visual_input_kind,
        }
    ).removeprefix("sha256:")
    return f"visual-neighbour-edge:{digest}"


def _edge_fingerprint(row: Mapping[str, object]) -> str:
    semantic = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "graph_fingerprint",
            "best_prototype_similarity",
            "supporting_prototype_pairs",
            "edge_fingerprint",
        }
    }
    encoded = canonical_semantic_bytes(semantic)
    preimage = bytearray(len(encoded).to_bytes(8, "big"))
    preimage.extend(encoded)
    preimage.extend(struct.pack("<d", float(row["best_prototype_similarity"])))
    pairs = row["supporting_prototype_pairs"]
    if not isinstance(pairs, Sequence):
        raise ValueError("supporting_prototype_pairs must be a sequence")
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("supporting prototype pair must be a mapping")
        for field in ("subject_prototype_id", "neighbour_prototype_id"):
            value = _required_text(pair.get(field), field=field).encode("utf-8")
            preimage.extend(len(value).to_bytes(8, "big"))
            preimage.extend(value)
        preimage.extend(struct.pack("<d", float(pair["similarity"])))
    return "sha256:" + hashlib.sha256(preimage).hexdigest()


def _graph_fingerprint(
    *,
    graph_version: str,
    configuration_fingerprint: str,
    reference_prototypes_fingerprint: str,
    edge_fingerprints: Sequence[str],
) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": VISUAL_NEIGHBOUR_GRAPH_FINGERPRINT_SCHEMA_VERSION,
            "graph_version": graph_version,
            "algorithm_version": VISUAL_NEIGHBOUR_GRAPH_ALGORITHM_VERSION,
            "configuration_fingerprint": configuration_fingerprint,
            "reference_prototypes_fingerprint": reference_prototypes_fingerprint,
            "edge_fingerprints": list(edge_fingerprints),
        }
    )


def _sort_graph_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.sort(
        [
            "route",
            "visual_input_kind",
            "subject_accepted_taxon_key",
            "neighbour_rank",
            "neighbour_accepted_taxon_key",
            "edge_id",
        ]
    )


def _cosine_similarity(left: object, right: object) -> float:
    if not isinstance(left, Sequence) or not isinstance(right, Sequence):
        raise ValueError("prototype embeddings must be sequences")
    similarity = fsum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    return min(1.0, max(-1.0, similarity))


def _single_text(frame: pl.DataFrame, field: str) -> str:
    values = frame[field].unique().to_list()
    if len(values) != 1:
        raise ValueError(f"visual-neighbour graph has mixed {field} values")
    return _required_text(values[0], field=field)


def _single_sha256(
    frame: pl.DataFrame,
    field: str,
    expected: str | None = None,
) -> str:
    value = _single_text(frame, field)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 fingerprint")
    if expected is not None and value != _sha256(expected, field=f"expected {field}"):
        raise ValueError(f"visual-neighbour graph {field} does not match expected")
    return value


def _single_positive_integer(frame: pl.DataFrame, field: str) -> int:
    values = frame[field].unique().to_list()
    if len(values) != 1:
        raise ValueError(f"visual-neighbour graph has mixed {field} values")
    return _positive_integer(values[0], field=field)


def _single_finite_float(frame: pl.DataFrame, field: str) -> float:
    values = frame[field].unique().to_list()
    if len(values) != 1:
        raise ValueError(f"visual-neighbour graph has mixed {field} values")
    return _finite_float(values[0], field=field)


def _bounded_similarity(value: object, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [-1, 1]")
    return result


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 fingerprint")
    return value


def _log_event(event: str, **fields: object) -> None:
    _LOGGER.info(
        "%s",
        json.dumps(
            {"event": event, **fields},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
    )


__all__ = [
    "VISUAL_NEIGHBOUR_GRAPH_ALGORITHM_VERSION",
    "VISUAL_NEIGHBOUR_SPECIES_FILE",
    "VISUAL_NEIGHBOUR_SPECIES_SCHEMA_VERSION",
    "VisualNeighbourGraphConfig",
    "build_visual_neighbour_species",
    "load_visual_neighbour_species",
    "validate_visual_neighbour_species",
    "visual_neighbour_species_schema",
    "write_visual_neighbour_species",
]
