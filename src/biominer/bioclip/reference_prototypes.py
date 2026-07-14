"""Deterministic SimpleShot-style prototypes over frozen support embeddings."""

from __future__ import annotations

from array import array
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
from math import fsum, isfinite, sqrt
from pathlib import Path
import re
import struct

import polars as pl

from biominer.bioclip.reference_embeddings import (
    load_reference_embeddings,
    reference_embeddings_artifact_fingerprint,
    validate_reference_embeddings,
)
from biominer.common.semantic_hash import (
    canonical_semantic_bytes,
    canonical_semantic_fingerprint,
)
from biominer.references.readiness import REFERENCE_ROUTES
from biominer.references.schemas import (
    REFERENCE_LIFE_STAGES,
    REFERENCE_VIEWS,
    REFERENCE_VISUAL_DOMAINS,
)
from biominer.storage.parquet import write_parquet
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
)


REFERENCE_PROTOTYPES_SCHEMA_VERSION = "reference-prototypes-v1.0.0"
REFERENCE_PROTOTYPES_FILE = "reference_prototypes.parquet"
REFERENCE_CENTERING_CONTEXT_SCHEMA_VERSION = "reference-centering-context-v1"
REFERENCE_PROTOTYPE_GROUP_SCHEMA_VERSION = "reference-prototype-group-v1"
REFERENCE_PROTOTYPE_ID_SCHEMA_VERSION = "reference-prototype-id-v1"

PROTOTYPE_METHOD_NORMALIZED_MEAN = "normalized_mean"
PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED = "simpleshot_mean_centered"
REFERENCE_PROTOTYPE_METHODS = frozenset(
    {
        PROTOTYPE_METHOD_NORMALIZED_MEAN,
        PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED,
    }
)
PROTOTYPE_SCOPE_GLOBAL = "global"
PROTOTYPE_SCOPE_REGIONAL = "regional"
REFERENCE_PROTOTYPE_SCOPE_TYPES = frozenset(
    {PROTOTYPE_SCOPE_GLOBAL, PROTOTYPE_SCOPE_REGIONAL}
)
PROTOTYPE_GLOBAL_GEO_CLUSTER_ID = "all"
PROTOTYPE_AGGREGATE_VIEW = "all"
DEFAULT_BALANCED_SAMPLING_SEED = 42

_VISUAL_INPUT_KINDS = frozenset(
    {
        RAW_FULL_IMAGE_KIND,
        FOCUSED_FULL_FRAME_KIND,
        MASKED_FULL_FRAME_KIND,
        MULTI_OBJECT_FULL_FRAME_KIND,
    }
)
_VISUAL_INPUT_KIND_ORDER = {
    RAW_FULL_IMAGE_KIND: 0,
    FOCUSED_FULL_FRAME_KIND: 1,
    MASKED_FULL_FRAME_KIND: 2,
    MULTI_OBJECT_FULL_FRAME_KIND: 3,
}
_PROTOTYPE_METHOD_ORDER = {
    PROTOTYPE_METHOD_NORMALIZED_MEAN: 0,
    PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED: 1,
}
_PROTOTYPE_SCOPE_ORDER = {
    PROTOTYPE_SCOPE_GLOBAL: 0,
    PROTOTYPE_SCOPE_REGIONAL: 1,
}
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UNIT_NORM_TOLERANCE = 1e-5
_ZERO_NORM_EPSILON = 1e-12
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReferenceCenteringContext:
    """Balanced route/input mean used for SimpleShot support and query centering."""

    route: str
    visual_input_kind: str
    embedding_dimension: int
    balanced_sampling_seed: int
    species_count: int
    reference_count: int
    independent_observation_count: int
    selected_observation_ids: tuple[str, ...]
    selected_observation_fingerprints: tuple[str, ...]
    mean_embedding: tuple[float, ...]
    centering_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.route not in REFERENCE_ROUTES:
            raise ValueError(f"unsupported reference centering route: {self.route}")
        if self.visual_input_kind not in _VISUAL_INPUT_KINDS:
            raise ValueError(
                "unsupported reference centering visual input kind: "
                f"{self.visual_input_kind}"
            )
        dimension = _positive_integer(
            self.embedding_dimension,
            field="centering embedding_dimension",
        )
        seed = _sampling_seed(self.balanced_sampling_seed)
        species_count = _positive_integer(
            self.species_count,
            field="centering species_count",
        )
        reference_count = _positive_integer(
            self.reference_count,
            field="centering reference_count",
        )
        observation_count = _positive_integer(
            self.independent_observation_count,
            field="centering independent_observation_count",
        )
        if reference_count < observation_count:
            raise ValueError(
                "centering reference_count cannot be smaller than independent observations"
            )
        selected = tuple(
            _required_text(value, field="selected_observation_ids")
            for value in self.selected_observation_ids
        )
        if len(selected) != observation_count or len(set(selected)) != len(selected):
            raise ValueError(
                "centering selected observations must be unique and match the count"
            )
        selected_fingerprints = tuple(
            _sha256(value, field="selected_observation_fingerprints")
            for value in self.selected_observation_fingerprints
        )
        if len(selected_fingerprints) != observation_count or len(
            set(selected_fingerprints)
        ) != len(selected_fingerprints):
            raise ValueError(
                "centering selected observation fingerprints must be unique and "
                "match the count"
            )
        mean = _finite_vector(
            self.mean_embedding,
            dimension=dimension,
            field="centering mean_embedding",
        )
        object.__setattr__(self, "embedding_dimension", dimension)
        object.__setattr__(self, "balanced_sampling_seed", seed)
        object.__setattr__(self, "species_count", species_count)
        object.__setattr__(self, "reference_count", reference_count)
        object.__setattr__(self, "independent_observation_count", observation_count)
        object.__setattr__(self, "selected_observation_ids", selected)
        object.__setattr__(
            self,
            "selected_observation_fingerprints",
            selected_fingerprints,
        )
        object.__setattr__(self, "mean_embedding", mean)
        expected_fingerprint = _centering_context_fingerprint(
            route=self.route,
            visual_input_kind=self.visual_input_kind,
            embedding_dimension=dimension,
            balanced_sampling_seed=seed,
            species_count=species_count,
            reference_count=reference_count,
            selected_observation_ids=selected,
            selected_observation_fingerprints=selected_fingerprints,
            mean_embedding=mean,
        )
        if (
            self.centering_fingerprint is not None
            and _sha256(
                self.centering_fingerprint,
                field="centering_fingerprint",
            )
            != expected_fingerprint
        ):
            raise ValueError("centering_fingerprint does not match its context")
        object.__setattr__(
            self,
            "centering_fingerprint",
            expected_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class _ObservationEmbedding:
    accepted_taxon_key: str
    scientific_name: str
    geo_cluster_id: str
    life_stage: str
    visual_domain: str
    route: str
    visual_input_kind: str
    reference_observation_id: str
    reference_count: int
    embedding: tuple[float, ...]
    contributor_fingerprints: tuple[str, ...]
    observation_fingerprint: str


@dataclass(frozen=True, slots=True)
class _PrototypeGroup:
    cluster_scope_type: str
    geo_cluster_id: str
    observations: tuple[_ObservationEmbedding, ...]


def reference_prototypes_schema(
    embedding_dimension: int,
) -> dict[str, pl.DataType]:
    dimension = _positive_integer(
        embedding_dimension,
        field="prototype embedding_dimension",
    )
    return {
        "schema_version": pl.String,
        "prototype_id": pl.String,
        "accepted_taxon_key": pl.String,
        "species": pl.String,
        "cluster_scope_type": pl.String,
        "geo_cluster_id": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "view": pl.String,
        "route": pl.String,
        "visual_input_kind": pl.String,
        "prototype_method": pl.String,
        "prototype_group_id": pl.String,
        "reference_count": pl.UInt32,
        "independent_observation_count": pl.UInt32,
        "balanced_sampling_seed": pl.UInt64,
        "mean_centered": pl.Boolean,
        "centering_fingerprint": pl.String,
        "embedding_dimension": pl.UInt32,
        "embedding": pl.Array(pl.Float32, dimension),
        "embedding_norm": pl.Float64,
        "model_fingerprint": pl.String,
        "reference_embedding_fingerprint": pl.String,
        "support_manifest_fingerprint": pl.String,
        "prototype_fingerprint": pl.String,
    }


def build_reference_centering_contexts(
    reference_embeddings: pl.DataFrame | str | Path,
    *,
    balanced_sampling_seed: int = DEFAULT_BALANCED_SAMPLING_SEED,
) -> tuple[ReferenceCenteringContext, ...]:
    """Compute class-balanced global means for every route/input contract."""

    frame = _reference_embedding_frame(reference_embeddings)
    support = _support_training_rows(frame)
    observations = _observation_embeddings(support)
    dimension = int(frame["embedding_dimension"][0])
    return _centering_contexts(
        observations,
        embedding_dimension=dimension,
        balanced_sampling_seed=balanced_sampling_seed,
    )


def mean_center_query_embedding(
    embedding: Sequence[float],
    context: ReferenceCenteringContext,
) -> tuple[float, ...]:
    """Subtract a persisted-context mean and return a unit Float32 query vector."""

    if not isinstance(context, ReferenceCenteringContext):
        raise TypeError("context must be a ReferenceCenteringContext")
    source = _finite_vector(
        embedding,
        dimension=context.embedding_dimension,
        field="query embedding",
    )
    source_norm = _vector_norm(source)
    if abs(source_norm - 1.0) > _UNIT_NORM_TOLERANCE:
        raise ValueError(
            "query embedding must be unit-normalized before mean-centering"
        )
    centered = tuple(
        value - mean for value, mean in zip(source, context.mean_embedding, strict=True)
    )
    stored, _ = _stored_unit_float32(centered, field="mean-centered query embedding")
    return stored


def build_reference_prototypes(
    reference_embeddings: pl.DataFrame | str | Path,
    *,
    balanced_sampling_seed: int = DEFAULT_BALANCED_SAMPLING_SEED,
    include_mean_centered: bool = True,
) -> pl.DataFrame:
    """Build raw and SimpleShot species centroids at global/regional scopes."""

    if not isinstance(include_mean_centered, bool):
        raise TypeError("include_mean_centered must be a boolean")
    seed = _sampling_seed(balanced_sampling_seed)
    frame = _reference_embedding_frame(reference_embeddings)
    support = _support_training_rows(frame)
    observations = _observation_embeddings(support)
    dimension = int(frame["embedding_dimension"][0])
    model_fingerprint = _single_sha256(frame, "model_fingerprint")
    support_manifest_fingerprint = _single_sha256(
        frame,
        "support_manifest_fingerprint",
    )
    embedding_artifact_fingerprint = reference_embeddings_artifact_fingerprint(support)
    contexts = _centering_contexts(
        observations,
        embedding_dimension=dimension,
        balanced_sampling_seed=seed,
    )
    contexts_by_key = {
        (context.route, context.visual_input_kind): context for context in contexts
    }
    _log_event(
        "reference_prototype_build_started",
        input_row_count=frame.height,
        support_train_row_count=support.height,
        independent_observation_count=len(observations),
        centering_context_count=len(contexts),
        balanced_sampling_seed=seed,
        include_mean_centered=include_mean_centered,
    )

    rows: list[dict[str, object]] = []
    skipped_centered_groups = 0
    for group in _prototype_groups(observations):
        group_id = _prototype_group_id(group)
        raw_embedding, raw_norm = _stored_unit_float32(
            _mean_vector([item.embedding for item in group.observations]),
            field="reference prototype",
        )
        rows.append(
            _prototype_row(
                group,
                prototype_group_id=group_id,
                prototype_method=PROTOTYPE_METHOD_NORMALIZED_MEAN,
                embedding=raw_embedding,
                embedding_norm=raw_norm,
                balanced_sampling_seed=None,
                centering_fingerprint=None,
                model_fingerprint=model_fingerprint,
                reference_embedding_fingerprint=embedding_artifact_fingerprint,
                support_manifest_fingerprint=support_manifest_fingerprint,
            )
        )
        if not include_mean_centered:
            continue
        first_observation = group.observations[0]
        context = contexts_by_key[
            (first_observation.route, first_observation.visual_input_kind)
        ]
        if context.species_count < 2:
            skipped_centered_groups += 1
            continue
        centered_support = [
            _mean_center_support_embedding(item.embedding, context.mean_embedding)
            for item in group.observations
        ]
        try:
            centered_embedding, centered_norm = _stored_unit_float32(
                _mean_vector(centered_support),
                field="mean-centered reference prototype",
            )
        except ValueError as exc:
            if "non-zero" not in str(exc):
                raise
            skipped_centered_groups += 1
            continue
        rows.append(
            _prototype_row(
                group,
                prototype_group_id=group_id,
                prototype_method=PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED,
                embedding=centered_embedding,
                embedding_norm=centered_norm,
                balanced_sampling_seed=seed,
                centering_fingerprint=context.centering_fingerprint,
                model_fingerprint=model_fingerprint,
                reference_embedding_fingerprint=embedding_artifact_fingerprint,
                support_manifest_fingerprint=support_manifest_fingerprint,
            )
        )

    result = _sort_prototype_frame(
        pl.DataFrame(
            rows,
            schema=reference_prototypes_schema(dimension),
            orient="row",
            strict=True,
        )
    )
    validate_reference_prototypes(result)
    _log_event(
        "reference_prototype_build_completed",
        input_row_count=frame.height,
        output_row_count=result.height,
        global_prototype_count=result.filter(
            pl.col("cluster_scope_type") == PROTOTYPE_SCOPE_GLOBAL
        ).height,
        regional_prototype_count=result.filter(
            pl.col("cluster_scope_type") == PROTOTYPE_SCOPE_REGIONAL
        ).height,
        mean_centered_prototype_count=result.filter(pl.col("mean_centered")).height,
        skipped_mean_centered_group_count=skipped_centered_groups,
        artifact_fingerprint=reference_prototypes_artifact_fingerprint(result),
    )
    return result


def validate_reference_prototypes(
    frame: pl.DataFrame,
    *,
    expected_model_fingerprint: str | None = None,
    expected_reference_embedding_fingerprint: str | None = None,
    expected_support_manifest_fingerprint: str | None = None,
) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("reference prototypes must be a Polars DataFrame")
    if frame.columns != list(reference_prototypes_schema(1)):
        raise ValueError("reference prototypes physical schema mismatch")
    if frame.is_empty():
        raise ValueError("reference prototypes artifact must not be empty")
    dimensions = frame["embedding_dimension"].unique().to_list()
    if len(dimensions) != 1:
        raise ValueError("reference prototypes artifact has mixed dimensions")
    dimension = _positive_integer(
        dimensions[0],
        field="prototype embedding_dimension",
    )
    if dict(frame.schema) != reference_prototypes_schema(dimension):
        raise ValueError("reference prototypes physical schema mismatch")
    if not frame.equals(_sort_prototype_frame(frame)):
        raise ValueError("reference prototypes are not deterministically sorted")
    if frame["prototype_id"].n_unique() != frame.height:
        raise ValueError("reference prototypes contain duplicate prototype IDs")
    if frame["prototype_fingerprint"].n_unique() != frame.height:
        raise ValueError("reference prototypes contain duplicate fingerprints")

    _single_sha256(frame, "model_fingerprint", expected_model_fingerprint)
    _single_sha256(
        frame,
        "reference_embedding_fingerprint",
        expected_reference_embedding_fingerprint,
    )
    _single_sha256(
        frame,
        "support_manifest_fingerprint",
        expected_support_manifest_fingerprint,
    )
    group_methods: dict[str, set[str]] = defaultdict(set)
    group_identity: dict[str, tuple[object, ...]] = {}
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != REFERENCE_PROTOTYPES_SCHEMA_VERSION:
            raise ValueError("unsupported reference prototype schema version")
        for field in (
            "accepted_taxon_key",
            "species",
            "geo_cluster_id",
            "life_stage",
            "visual_domain",
            "view",
            "route",
            "visual_input_kind",
            "prototype_method",
        ):
            _required_text(row[field], field=field)
        scope = str(row["cluster_scope_type"])
        if scope not in REFERENCE_PROTOTYPE_SCOPE_TYPES:
            raise ValueError(f"unsupported reference prototype scope: {scope}")
        cluster_id = str(row["geo_cluster_id"])
        if (scope == PROTOTYPE_SCOPE_GLOBAL) != (
            cluster_id == PROTOTYPE_GLOBAL_GEO_CLUSTER_ID
        ):
            raise ValueError("reference prototype cluster scope is inconsistent")
        if row["life_stage"] not in REFERENCE_LIFE_STAGES:
            raise ValueError("reference prototype life_stage is invalid")
        if row["visual_domain"] not in REFERENCE_VISUAL_DOMAINS:
            raise ValueError("reference prototype visual_domain is invalid")
        if row["view"] not in {*REFERENCE_VIEWS, PROTOTYPE_AGGREGATE_VIEW}:
            raise ValueError("reference prototype view is invalid")
        if row["route"] not in REFERENCE_ROUTES:
            raise ValueError("reference prototype route is invalid")
        if row["visual_input_kind"] not in _VISUAL_INPUT_KINDS:
            raise ValueError("reference prototype visual_input_kind is invalid")
        method = str(row["prototype_method"])
        if method not in REFERENCE_PROTOTYPE_METHODS:
            raise ValueError(f"unsupported reference prototype method: {method}")
        expected_centered = method == PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED
        if bool(row["mean_centered"]) != expected_centered:
            raise ValueError("reference prototype mean_centered flag is inconsistent")
        seed = row["balanced_sampling_seed"]
        centering_fingerprint = row["centering_fingerprint"]
        if expected_centered:
            parsed_seed: int | None = _sampling_seed(seed)
            parsed_centering: str | None = _sha256(
                centering_fingerprint,
                field="centering_fingerprint",
            )
        else:
            if seed is not None or centering_fingerprint is not None:
                raise ValueError(
                    "raw reference prototypes cannot declare centering provenance"
                )
            parsed_seed = None
            parsed_centering = None
        reference_count = _positive_integer(
            row["reference_count"],
            field="reference_count",
        )
        observation_count = _positive_integer(
            row["independent_observation_count"],
            field="independent_observation_count",
        )
        if reference_count < observation_count:
            raise ValueError(
                "prototype reference_count cannot be smaller than independent observations"
            )
        vector = _finite_vector(
            row["embedding"],
            dimension=dimension,
            field="prototype embedding",
        )
        norm = _finite_positive_float(row["embedding_norm"], field="embedding_norm")
        actual_norm = _vector_norm(vector)
        if abs(actual_norm - norm) > _UNIT_NORM_TOLERANCE:
            raise ValueError(
                "prototype embedding_norm does not match stored Float32 values"
            )
        if abs(actual_norm - 1.0) > _UNIT_NORM_TOLERANCE:
            raise ValueError("reference prototype must be unit-normalized")
        group_id = _sha256(row["prototype_group_id"], field="prototype_group_id")
        expected_id = _prototype_id(
            prototype_group_id=group_id,
            prototype_method=method,
            balanced_sampling_seed=parsed_seed,
            centering_fingerprint=parsed_centering,
        )
        if row["prototype_id"] != expected_id:
            raise ValueError("reference prototype ID is invalid")
        if row["prototype_fingerprint"] != _prototype_fingerprint(row):
            raise ValueError("reference prototype fingerprint is invalid")
        methods = group_methods[group_id]
        if method in methods:
            raise ValueError("reference prototype group repeats a method")
        methods.add(method)
        identity = (
            row["accepted_taxon_key"],
            row["species"],
            row["cluster_scope_type"],
            row["geo_cluster_id"],
            row["life_stage"],
            row["visual_domain"],
            row["view"],
            row["route"],
            row["visual_input_kind"],
            reference_count,
            observation_count,
        )
        previous = group_identity.setdefault(group_id, identity)
        if previous != identity:
            raise ValueError("reference prototype group identity is inconsistent")
    for methods in group_methods.values():
        if PROTOTYPE_METHOD_NORMALIZED_MEAN not in methods:
            raise ValueError("reference prototype group lacks its raw centroid")


def reference_prototypes_artifact_fingerprint(frame: pl.DataFrame) -> str:
    validate_reference_prototypes(frame)
    digest = hashlib.sha256()
    for fingerprint in frame["prototype_fingerprint"].to_list():
        encoded = str(fingerprint).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return "sha256:" + digest.hexdigest()


def write_reference_prototypes(
    frame: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    validate_reference_prototypes(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= REFERENCE_PROTOTYPES_FILE
    written = write_parquet(frame, destination, overwrite=overwrite)
    loaded = pl.read_parquet(written)
    validate_reference_prototypes(loaded)
    if not frame.equals(loaded):
        raise ValueError("reference prototypes Parquet round-trip mismatch")
    _log_event(
        "reference_prototypes_written",
        artifact_path=str(written),
        row_count=frame.height,
        byte_count=written.stat().st_size,
        artifact_fingerprint=reference_prototypes_artifact_fingerprint(frame),
    )
    return written


def load_reference_prototypes(
    path: str | Path,
    *,
    expected_model_fingerprint: str | None = None,
    expected_reference_embedding_fingerprint: str | None = None,
    expected_support_manifest_fingerprint: str | None = None,
) -> pl.DataFrame:
    source = Path(path)
    if source.is_dir():
        source /= REFERENCE_PROTOTYPES_FILE
    frame = pl.read_parquet(source)
    validate_reference_prototypes(
        frame,
        expected_model_fingerprint=expected_model_fingerprint,
        expected_reference_embedding_fingerprint=(
            expected_reference_embedding_fingerprint
        ),
        expected_support_manifest_fingerprint=expected_support_manifest_fingerprint,
    )
    return frame


def _reference_embedding_frame(
    source: pl.DataFrame | str | Path,
) -> pl.DataFrame:
    if isinstance(source, pl.DataFrame):
        validate_reference_embeddings(source)
        return source
    return load_reference_embeddings(source)


def _support_training_rows(frame: pl.DataFrame) -> pl.DataFrame:
    support = frame.filter(pl.col("support_split") == "support_train")
    if support.is_empty():
        raise ValueError("reference prototypes require support_train embeddings")
    return support


def _observation_embeddings(
    support: pl.DataFrame,
) -> tuple[_ObservationEmbedding, ...]:
    group_fields = (
        "accepted_taxon_key",
        "scientific_name",
        "geo_cluster_id",
        "life_stage",
        "visual_domain",
        "route",
        "visual_input_kind",
        "reference_observation_id",
    )
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    observation_identity: dict[tuple[str, str], tuple[str, ...]] = {}
    for row in support.iter_rows(named=True):
        key = tuple(str(row[field]) for field in group_fields)
        identity_key = (
            str(row["reference_observation_id"]),
            str(row["visual_input_kind"]),
        )
        semantic_identity = key[:-1]
        previous = observation_identity.setdefault(identity_key, semantic_identity)
        if previous != semantic_identity:
            raise ValueError(
                "reference observation crosses taxon, route, domain, or geographic identity"
            )
        grouped[key].append(row)

    observations: list[_ObservationEmbedding] = []
    for key, rows in grouped.items():
        rows.sort(key=lambda row: str(row["embedding_fingerprint"]))
        dimension = int(rows[0]["embedding_dimension"])
        mean = _mean_vector(
            [
                _finite_vector(
                    row["embedding"],
                    dimension=dimension,
                    field="reference embedding",
                )
                for row in rows
            ]
        )
        embedding = _unit_vector64(mean, field="independent observation embedding")
        contributor_fingerprints = tuple(
            str(row["embedding_fingerprint"]) for row in rows
        )
        payload = {
            "accepted_taxon_key": key[0],
            "scientific_name": key[1],
            "geo_cluster_id": key[2],
            "life_stage": key[3],
            "visual_domain": key[4],
            "route": key[5],
            "visual_input_kind": key[6],
            "reference_observation_id": key[7],
            "contributor_fingerprints": contributor_fingerprints,
            "embedding": embedding,
        }
        observations.append(
            _ObservationEmbedding(
                accepted_taxon_key=key[0],
                scientific_name=key[1],
                geo_cluster_id=key[2],
                life_stage=key[3],
                visual_domain=key[4],
                route=key[5],
                visual_input_kind=key[6],
                reference_observation_id=key[7],
                reference_count=len(rows),
                embedding=embedding,
                contributor_fingerprints=contributor_fingerprints,
                observation_fingerprint=canonical_semantic_fingerprint(payload),
            )
        )
    return tuple(sorted(observations, key=_observation_sort_key))


def _centering_contexts(
    observations: Sequence[_ObservationEmbedding],
    *,
    embedding_dimension: int,
    balanced_sampling_seed: int,
) -> tuple[ReferenceCenteringContext, ...]:
    seed = _sampling_seed(balanced_sampling_seed)
    grouped: dict[tuple[str, str], list[_ObservationEmbedding]] = defaultdict(list)
    for item in observations:
        grouped[(item.route, item.visual_input_kind)].append(item)
    contexts: list[ReferenceCenteringContext] = []
    for (route, visual_input_kind), items in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], _VISUAL_INPUT_KIND_ORDER[item[0][1]]),
    ):
        by_species: dict[str, list[_ObservationEmbedding]] = defaultdict(list)
        for item in items:
            by_species[item.accepted_taxon_key].append(item)
        balanced_count = min(len(values) for values in by_species.values())
        selected: list[_ObservationEmbedding] = []
        for taxon_key in sorted(by_species):
            ranked = sorted(
                by_species[taxon_key],
                key=lambda item: _balanced_observation_rank(
                    item,
                    balanced_sampling_seed=seed,
                ),
            )
            selected.extend(ranked[:balanced_count])
        selected.sort(key=_observation_sort_key)
        mean = _mean_vector([item.embedding for item in selected])
        contexts.append(
            ReferenceCenteringContext(
                route=route,
                visual_input_kind=visual_input_kind,
                embedding_dimension=embedding_dimension,
                balanced_sampling_seed=seed,
                species_count=len(by_species),
                reference_count=sum(item.reference_count for item in selected),
                independent_observation_count=len(selected),
                selected_observation_ids=tuple(
                    item.reference_observation_id for item in selected
                ),
                selected_observation_fingerprints=tuple(
                    item.observation_fingerprint for item in selected
                ),
                mean_embedding=mean,
            )
        )
    return tuple(contexts)


def _prototype_groups(
    observations: Sequence[_ObservationEmbedding],
) -> tuple[_PrototypeGroup, ...]:
    groups: dict[tuple[str, ...], list[_ObservationEmbedding]] = defaultdict(list)
    for item in observations:
        base = (
            item.accepted_taxon_key,
            item.scientific_name,
            item.life_stage,
            item.visual_domain,
            item.route,
            item.visual_input_kind,
        )
        groups[(*base, PROTOTYPE_SCOPE_GLOBAL, PROTOTYPE_GLOBAL_GEO_CLUSTER_ID)].append(
            item
        )
        groups[(*base, PROTOTYPE_SCOPE_REGIONAL, item.geo_cluster_id)].append(item)
    result: list[_PrototypeGroup] = []
    for key, values in sorted(groups.items(), key=lambda item: item[0]):
        result.append(
            _PrototypeGroup(
                cluster_scope_type=key[-2],
                geo_cluster_id=key[-1],
                observations=tuple(sorted(values, key=_observation_sort_key)),
            )
        )
    return tuple(result)


def _prototype_group_id(group: _PrototypeGroup) -> str:
    first = group.observations[0]
    return canonical_semantic_fingerprint(
        {
            "schema_version": REFERENCE_PROTOTYPE_GROUP_SCHEMA_VERSION,
            "accepted_taxon_key": first.accepted_taxon_key,
            "species": first.scientific_name,
            "cluster_scope_type": group.cluster_scope_type,
            "geo_cluster_id": group.geo_cluster_id,
            "life_stage": first.life_stage,
            "visual_domain": first.visual_domain,
            "view": PROTOTYPE_AGGREGATE_VIEW,
            "route": first.route,
            "visual_input_kind": first.visual_input_kind,
            "observation_fingerprints": [
                item.observation_fingerprint for item in group.observations
            ],
        }
    )


def _prototype_row(
    group: _PrototypeGroup,
    *,
    prototype_group_id: str,
    prototype_method: str,
    embedding: tuple[float, ...],
    embedding_norm: float,
    balanced_sampling_seed: int | None,
    centering_fingerprint: str | None,
    model_fingerprint: str,
    reference_embedding_fingerprint: str,
    support_manifest_fingerprint: str,
) -> dict[str, object]:
    first = group.observations[0]
    row: dict[str, object] = {
        "schema_version": REFERENCE_PROTOTYPES_SCHEMA_VERSION,
        "prototype_id": _prototype_id(
            prototype_group_id=prototype_group_id,
            prototype_method=prototype_method,
            balanced_sampling_seed=balanced_sampling_seed,
            centering_fingerprint=centering_fingerprint,
        ),
        "accepted_taxon_key": first.accepted_taxon_key,
        "species": first.scientific_name,
        "cluster_scope_type": group.cluster_scope_type,
        "geo_cluster_id": group.geo_cluster_id,
        "life_stage": first.life_stage,
        "visual_domain": first.visual_domain,
        "view": PROTOTYPE_AGGREGATE_VIEW,
        "route": first.route,
        "visual_input_kind": first.visual_input_kind,
        "prototype_method": prototype_method,
        "prototype_group_id": prototype_group_id,
        "reference_count": sum(item.reference_count for item in group.observations),
        "independent_observation_count": len(group.observations),
        "balanced_sampling_seed": balanced_sampling_seed,
        "mean_centered": (
            prototype_method == PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED
        ),
        "centering_fingerprint": centering_fingerprint,
        "embedding_dimension": len(embedding),
        "embedding": list(embedding),
        "embedding_norm": embedding_norm,
        "model_fingerprint": model_fingerprint,
        "reference_embedding_fingerprint": reference_embedding_fingerprint,
        "support_manifest_fingerprint": support_manifest_fingerprint,
        "prototype_fingerprint": "",
    }
    row["prototype_fingerprint"] = _prototype_fingerprint(row)
    return row


def _prototype_id(
    *,
    prototype_group_id: str,
    prototype_method: str,
    balanced_sampling_seed: int | None,
    centering_fingerprint: str | None,
) -> str:
    digest = canonical_semantic_fingerprint(
        {
            "schema_version": REFERENCE_PROTOTYPE_ID_SCHEMA_VERSION,
            "prototype_group_id": prototype_group_id,
            "prototype_method": prototype_method,
            "balanced_sampling_seed": balanced_sampling_seed,
            "centering_fingerprint": centering_fingerprint,
        }
    ).removeprefix("sha256:")
    return f"reference-prototype:{digest}"


def _prototype_fingerprint_preimage(row: Mapping[str, object]) -> bytes:
    vector = tuple(float(value) for value in row["embedding"])
    semantic = {
        key: value
        for key, value in row.items()
        if key not in {"embedding", "embedding_norm", "prototype_fingerprint"}
    }
    encoded = canonical_semantic_bytes(semantic)
    preimage = bytearray(len(encoded).to_bytes(8, "big"))
    preimage.extend(encoded)
    preimage.extend(struct.pack("<d", float(row["embedding_norm"])))
    for value in vector:
        preimage.extend(struct.pack("<f", value))
    return bytes(preimage)


def _prototype_fingerprint(row: Mapping[str, object]) -> str:
    return "sha256:" + hashlib.sha256(_prototype_fingerprint_preimage(row)).hexdigest()


def _sort_prototype_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return (
        frame.with_columns(
            pl.col("cluster_scope_type")
            .replace_strict(_PROTOTYPE_SCOPE_ORDER, return_dtype=pl.UInt8)
            .alias("__scope_order"),
            pl.col("visual_input_kind")
            .replace_strict(_VISUAL_INPUT_KIND_ORDER, return_dtype=pl.UInt8)
            .alias("__visual_input_order"),
            pl.col("prototype_method")
            .replace_strict(_PROTOTYPE_METHOD_ORDER, return_dtype=pl.UInt8)
            .alias("__method_order"),
        )
        .sort(
            [
                "route",
                "species",
                "__scope_order",
                "geo_cluster_id",
                "life_stage",
                "visual_domain",
                "__visual_input_order",
                "__method_order",
                "prototype_id",
            ]
        )
        .drop("__scope_order", "__visual_input_order", "__method_order")
    )


def _balanced_observation_rank(
    item: _ObservationEmbedding,
    *,
    balanced_sampling_seed: int,
) -> str:
    return canonical_semantic_fingerprint(
        {
            "balanced_sampling_seed": balanced_sampling_seed,
            "accepted_taxon_key": item.accepted_taxon_key,
            "reference_observation_id": item.reference_observation_id,
            "route": item.route,
            "visual_input_kind": item.visual_input_kind,
        }
    )


def _centering_context_fingerprint(
    *,
    route: str,
    visual_input_kind: str,
    embedding_dimension: int,
    balanced_sampling_seed: int,
    species_count: int,
    reference_count: int,
    selected_observation_ids: Sequence[str],
    selected_observation_fingerprints: Sequence[str],
    mean_embedding: Sequence[float],
) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": REFERENCE_CENTERING_CONTEXT_SCHEMA_VERSION,
            "route": route,
            "visual_input_kind": visual_input_kind,
            "embedding_dimension": embedding_dimension,
            "balanced_sampling_seed": balanced_sampling_seed,
            "species_count": species_count,
            "reference_count": reference_count,
            "selected_observations": [
                {
                    "reference_observation_id": observation_id,
                    "observation_fingerprint": observation_fingerprint,
                }
                for observation_id, observation_fingerprint in zip(
                    selected_observation_ids,
                    selected_observation_fingerprints,
                    strict=True,
                )
            ],
            "mean_embedding": tuple(mean_embedding),
        }
    )


def _mean_center_support_embedding(
    embedding: Sequence[float],
    mean: Sequence[float],
) -> tuple[float, ...]:
    centered = tuple(
        value - offset for value, offset in zip(embedding, mean, strict=True)
    )
    norm = _vector_norm(centered)
    if norm <= _ZERO_NORM_EPSILON:
        return tuple(0.0 for _ in centered)
    return tuple(value / norm for value in centered)


def _mean_vector(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        raise ValueError("cannot calculate a mean from no embeddings")
    dimension = len(vectors[0])
    if dimension <= 0 or any(len(vector) != dimension for vector in vectors):
        raise ValueError("embedding mean requires one consistent positive dimension")
    count = len(vectors)
    return tuple(
        fsum(vector[index] for vector in vectors) / count for index in range(dimension)
    )


def _unit_vector64(values: Sequence[float], *, field: str) -> tuple[float, ...]:
    source = tuple(float(value) for value in values)
    norm = _vector_norm(source)
    if norm <= _ZERO_NORM_EPSILON:
        raise ValueError(f"{field} must have non-zero norm")
    return tuple(value / norm for value in source)


def _stored_unit_float32(
    values: Sequence[float],
    *,
    field: str,
) -> tuple[tuple[float, ...], float]:
    normalized = _unit_vector64(values, field=field)
    try:
        stored = tuple(float(value) for value in array("f", normalized))
    except OverflowError as exc:
        raise ValueError(f"{field} must contain finite Float32 values") from exc
    if any(not isfinite(value) for value in stored):
        raise ValueError(f"{field} must contain finite Float32 values")
    norm = _vector_norm(stored)
    if norm <= _ZERO_NORM_EPSILON:
        raise ValueError(f"{field} must have non-zero norm")
    if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
        raise ValueError(f"{field} Float32 representation is not unit-normalized")
    return stored, norm


def _finite_vector(
    values: object,
    *,
    dimension: int,
    field: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ValueError(f"{field} must be an embedding sequence")
    result: list[float] = []
    for raw in values:
        if isinstance(raw, bool):
            raise ValueError(f"{field} must contain finite numeric values")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain finite numeric values") from exc
        if not isfinite(value):
            raise ValueError(f"{field} must contain finite numeric values")
        result.append(value)
    if len(result) != dimension:
        raise ValueError(f"{field} dimension mismatch")
    return tuple(result)


def _vector_norm(values: Sequence[float]) -> float:
    return sqrt(fsum(value * value for value in values))


def _observation_sort_key(item: _ObservationEmbedding) -> tuple[object, ...]:
    return (
        item.route,
        _VISUAL_INPUT_KIND_ORDER[item.visual_input_kind],
        item.accepted_taxon_key,
        item.geo_cluster_id,
        item.reference_observation_id,
        item.observation_fingerprint,
    )


def _single_sha256(
    frame: pl.DataFrame,
    field: str,
    expected: str | None = None,
) -> str:
    values = frame[field].unique().to_list()
    if len(values) != 1:
        raise ValueError(f"reference artifact has mixed {field} values")
    value = _sha256(values[0], field=field)
    if expected is not None and value != _sha256(expected, field=f"expected {field}"):
        raise ValueError(f"reference artifact {field} does not match expected identity")
    return value


def _sampling_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("balanced_sampling_seed must be an integer")
    if not 0 <= value <= 18_446_744_073_709_551_615:
        raise ValueError("balanced_sampling_seed must fit an unsigned 64-bit integer")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite and positive") from exc
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _required_text(value: object, *, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field} must be non-empty")
    return result


def _sha256(value: object, *, field: str) -> str:
    result = str(value or "")
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 fingerprint")
    return result


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
    "DEFAULT_BALANCED_SAMPLING_SEED",
    "PROTOTYPE_AGGREGATE_VIEW",
    "PROTOTYPE_GLOBAL_GEO_CLUSTER_ID",
    "PROTOTYPE_METHOD_NORMALIZED_MEAN",
    "PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED",
    "PROTOTYPE_SCOPE_GLOBAL",
    "PROTOTYPE_SCOPE_REGIONAL",
    "REFERENCE_PROTOTYPES_FILE",
    "REFERENCE_PROTOTYPES_SCHEMA_VERSION",
    "REFERENCE_PROTOTYPE_METHODS",
    "REFERENCE_PROTOTYPE_SCOPE_TYPES",
    "ReferenceCenteringContext",
    "build_reference_centering_contexts",
    "build_reference_prototypes",
    "load_reference_prototypes",
    "mean_center_query_embedding",
    "reference_prototypes_artifact_fingerprint",
    "reference_prototypes_schema",
    "validate_reference_prototypes",
    "write_reference_prototypes",
]
