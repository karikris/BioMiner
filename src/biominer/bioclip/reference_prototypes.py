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


REFERENCE_PROTOTYPES_SCHEMA_VERSION = "reference-prototypes-v2.0.0"
REFERENCE_PROTOTYPES_FILE = "reference_prototypes.parquet"
REFERENCE_CENTERING_CONTEXT_SCHEMA_VERSION = "reference-centering-context-v1"
REFERENCE_PROTOTYPE_GROUP_SCHEMA_VERSION = "reference-prototype-group-v2"
REFERENCE_PROTOTYPE_ID_SCHEMA_VERSION = "reference-prototype-id-v2"
MULTI_PROTOTYPE_CONFIG_SCHEMA_VERSION = "multi-prototype-config-v1"

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

PROTOTYPE_KIND_AGGREGATE = "aggregate"
PROTOTYPE_KIND_METADATA = "metadata"
PROTOTYPE_KIND_EMBEDDING_CLUSTER = "embedding_cluster"
REFERENCE_PROTOTYPE_KINDS = frozenset(
    {
        PROTOTYPE_KIND_AGGREGATE,
        PROTOTYPE_KIND_METADATA,
        PROTOTYPE_KIND_EMBEDDING_CLUSTER,
    }
)
EMBEDDING_CLUSTER_METHOD_AVERAGE_LINKAGE_COSINE = (
    "deterministic_average_linkage_cosine_v1"
)

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
_PROTOTYPE_KIND_ORDER = {
    PROTOTYPE_KIND_AGGREGATE: 0,
    PROTOTYPE_KIND_METADATA: 1,
    PROTOTYPE_KIND_EMBEDDING_CLUSTER: 2,
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
class MultiPrototypeConfig:
    """Deterministic metadata and within-species clustering policy."""

    minimum_metadata_observation_count: int = 2
    enable_embedding_clustering: bool = True
    minimum_clustering_observation_count: int = 8
    minimum_embedding_cluster_size: int = 3
    maximum_embedding_cluster_count: int = 4
    maximum_clustering_observation_count: int = 256
    cosine_distance_threshold: float = 0.20

    def __post_init__(self) -> None:
        metadata_minimum = _positive_integer(
            self.minimum_metadata_observation_count,
            field="minimum_metadata_observation_count",
        )
        if not isinstance(self.enable_embedding_clustering, bool):
            raise TypeError("enable_embedding_clustering must be a boolean")
        clustering_minimum = _positive_integer(
            self.minimum_clustering_observation_count,
            field="minimum_clustering_observation_count",
        )
        cluster_minimum = _positive_integer(
            self.minimum_embedding_cluster_size,
            field="minimum_embedding_cluster_size",
        )
        cluster_maximum = _positive_integer(
            self.maximum_embedding_cluster_count,
            field="maximum_embedding_cluster_count",
        )
        observation_maximum = _positive_integer(
            self.maximum_clustering_observation_count,
            field="maximum_clustering_observation_count",
        )
        if clustering_minimum < 2 * cluster_minimum:
            raise ValueError(
                "minimum_clustering_observation_count must permit two minimum-size clusters"
            )
        if cluster_maximum < 2:
            raise ValueError("maximum_embedding_cluster_count must be at least two")
        if observation_maximum < clustering_minimum:
            raise ValueError(
                "maximum_clustering_observation_count cannot be smaller than the minimum"
            )
        threshold = _finite_float(
            self.cosine_distance_threshold,
            field="cosine_distance_threshold",
        )
        if not 0.0 <= threshold <= 2.0:
            raise ValueError("cosine_distance_threshold must be in [0, 2]")
        object.__setattr__(
            self,
            "minimum_metadata_observation_count",
            metadata_minimum,
        )
        object.__setattr__(
            self,
            "minimum_clustering_observation_count",
            clustering_minimum,
        )
        object.__setattr__(
            self,
            "minimum_embedding_cluster_size",
            cluster_minimum,
        )
        object.__setattr__(
            self,
            "maximum_embedding_cluster_count",
            cluster_maximum,
        )
        object.__setattr__(
            self,
            "maximum_clustering_observation_count",
            observation_maximum,
        )
        object.__setattr__(self, "cosine_distance_threshold", threshold)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": MULTI_PROTOTYPE_CONFIG_SCHEMA_VERSION,
                "minimum_metadata_observation_count": (
                    self.minimum_metadata_observation_count
                ),
                "enable_embedding_clustering": self.enable_embedding_clustering,
                "minimum_clustering_observation_count": (
                    self.minimum_clustering_observation_count
                ),
                "minimum_embedding_cluster_size": (self.minimum_embedding_cluster_size),
                "maximum_embedding_cluster_count": (
                    self.maximum_embedding_cluster_count
                ),
                "maximum_clustering_observation_count": (
                    self.maximum_clustering_observation_count
                ),
                "cosine_distance_threshold": self.cosine_distance_threshold,
            }
        )


@dataclass(frozen=True, slots=True)
class ReferenceObservationEmbedding:
    """One equally weighted biological observation in one scoring contract."""

    accepted_taxon_key: str
    scientific_name: str
    geo_cluster_id: str
    life_stage: str
    visual_domain: str
    view: str
    route: str
    visual_input_kind: str
    reference_observation_id: str
    duplicate_group_ids: tuple[str, ...]
    reference_count: int
    embedding: tuple[float, ...]
    contributor_fingerprints: tuple[str, ...]
    observation_fingerprint: str


@dataclass(frozen=True, slots=True)
class _PrototypeGroup:
    cluster_scope_type: str
    geo_cluster_id: str
    observations: tuple[ReferenceObservationEmbedding, ...]
    prototype_kind: str = PROTOTYPE_KIND_AGGREGATE
    view: str = PROTOTYPE_AGGREGATE_VIEW
    metadata_group_id: str | None = None
    embedding_cluster_id: str | None = None
    clustering_config: MultiPrototypeConfig | None = None


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
        "prototype_kind": pl.String,
        "prototype_method": pl.String,
        "prototype_group_id": pl.String,
        "metadata_group_id": pl.String,
        "embedding_cluster_id": pl.String,
        "clustering_method": pl.String,
        "clustering_configuration_fingerprint": pl.String,
        "clustering_cosine_distance_threshold": pl.Float64,
        "clustering_minimum_metadata_observation_count": pl.UInt32,
        "clustering_minimum_observation_count": pl.UInt32,
        "clustering_minimum_cluster_size": pl.UInt32,
        "clustering_maximum_cluster_count": pl.UInt32,
        "clustering_maximum_observation_count": pl.UInt32,
        "member_observation_ids": pl.List(pl.String),
        "member_observation_fingerprints": pl.List(pl.String),
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


def aggregate_reference_observation_embeddings(
    reference_embeddings: pl.DataFrame | str | Path,
    *,
    preserve_view: bool = False,
) -> tuple[ReferenceObservationEmbedding, ...]:
    """Collapse support-training media to equally weighted observations."""

    if not isinstance(preserve_view, bool):
        raise TypeError("preserve_view must be a boolean")
    frame = _reference_embedding_frame(reference_embeddings)
    return _observation_embeddings(
        _support_training_rows(frame),
        preserve_view=preserve_view,
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
    multi_prototype_config: MultiPrototypeConfig | None = None,
) -> pl.DataFrame:
    """Build raw and SimpleShot species centroids at global/regional scopes."""

    if not isinstance(include_mean_centered, bool):
        raise TypeError("include_mean_centered must be a boolean")
    if multi_prototype_config is not None and not isinstance(
        multi_prototype_config, MultiPrototypeConfig
    ):
        raise TypeError("multi_prototype_config must be a MultiPrototypeConfig")
    seed = _sampling_seed(balanced_sampling_seed)
    frame = _reference_embedding_frame(reference_embeddings)
    support = _support_training_rows(frame)
    observations = _observation_embeddings(support)
    view_observations = (
        _observation_embeddings(support, preserve_view=True)
        if multi_prototype_config is not None
        else ()
    )
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
        multi_prototype_config_fingerprint=(
            multi_prototype_config.fingerprint
            if multi_prototype_config is not None
            else None
        ),
    )

    rows: list[dict[str, object]] = []
    skipped_centered_groups = 0
    groups = list(_prototype_groups(observations))
    if multi_prototype_config is not None:
        groups.extend(
            _multi_prototype_groups(
                view_observations,
                config=multi_prototype_config,
            )
        )
    groups.sort(key=_prototype_group_sort_key)
    for group in groups:
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
        metadata_prototype_count=result.filter(
            pl.col("prototype_kind") == PROTOTYPE_KIND_METADATA
        ).height,
        embedding_cluster_prototype_count=result.filter(
            pl.col("prototype_kind") == PROTOTYPE_KIND_EMBEDDING_CLUSTER
        ).height,
        skipped_mean_centered_group_count=skipped_centered_groups,
        artifact_fingerprint=reference_prototypes_artifact_fingerprint(result),
    )
    return result


def build_multi_reference_prototypes(
    reference_embeddings: pl.DataFrame | str | Path,
    *,
    config: MultiPrototypeConfig | None = None,
    balanced_sampling_seed: int = DEFAULT_BALANCED_SAMPLING_SEED,
    include_mean_centered: bool = True,
) -> pl.DataFrame:
    """Build aggregate, metadata-led, and within-species cluster prototypes."""

    effective_config = config or MultiPrototypeConfig()
    if not isinstance(effective_config, MultiPrototypeConfig):
        raise TypeError("config must be a MultiPrototypeConfig")
    return build_reference_prototypes(
        reference_embeddings,
        balanced_sampling_seed=balanced_sampling_seed,
        include_mean_centered=include_mean_centered,
        multi_prototype_config=effective_config,
    )


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
    metadata_group_identity: dict[str, tuple[object, ...]] = {}
    metadata_group_members: dict[str, frozenset[tuple[str, str]]] = {}
    cluster_parent_identity: dict[str, tuple[object, ...]] = {}
    cluster_group_members: dict[
        str,
        tuple[str, frozenset[tuple[str, str]]],
    ] = {}
    cluster_policy_by_parent: dict[str, MultiPrototypeConfig] = {}
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
            "prototype_kind",
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
        prototype_kind = str(row["prototype_kind"])
        if prototype_kind not in REFERENCE_PROTOTYPE_KINDS:
            raise ValueError(f"unsupported reference prototype kind: {prototype_kind}")
        member_ids = _sorted_unique_text_list(
            row["member_observation_ids"],
            field="member_observation_ids",
        )
        member_fingerprints = _sha256_list(
            row["member_observation_fingerprints"],
            field="member_observation_fingerprints",
        )
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
        if (
            len(member_ids) != observation_count
            or len(member_fingerprints) != observation_count
        ):
            raise ValueError(
                "prototype member observations must match the independent count"
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
        (
            metadata_group_id,
            embedding_cluster_id,
            clustering_fingerprint,
            clustering_config,
        ) = _validate_prototype_kind_fields(
            row,
            prototype_kind=prototype_kind,
            prototype_group_id=group_id,
            member_ids=member_ids,
            member_fingerprints=member_fingerprints,
        )
        if group_id != _prototype_group_id_from_row(
            row,
            member_fingerprints=member_fingerprints,
            metadata_group_id=metadata_group_id,
            embedding_cluster_id=embedding_cluster_id,
            clustering_configuration_fingerprint=clustering_fingerprint,
        ):
            raise ValueError("reference prototype group ID is invalid")
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
            prototype_kind,
            metadata_group_id,
            embedding_cluster_id,
            clustering_fingerprint,
            member_ids,
            member_fingerprints,
            reference_count,
            observation_count,
        )
        previous = group_identity.setdefault(group_id, identity)
        if previous != identity:
            raise ValueError("reference prototype group identity is inconsistent")
        relationship_identity = (
            row["accepted_taxon_key"],
            row["species"],
            row["cluster_scope_type"],
            row["geo_cluster_id"],
            row["life_stage"],
            row["visual_domain"],
            row["view"],
            row["route"],
            row["visual_input_kind"],
        )
        if prototype_kind == PROTOTYPE_KIND_METADATA:
            metadata_group_identity[group_id] = relationship_identity
            metadata_group_members[group_id] = frozenset(
                zip(member_ids, member_fingerprints, strict=True)
            )
        elif prototype_kind == PROTOTYPE_KIND_EMBEDDING_CLUSTER:
            if metadata_group_id is None:
                raise AssertionError("validated cluster lacks its metadata parent")
            if clustering_config is None:
                raise AssertionError("validated cluster lacks its clustering policy")
            if observation_count < clustering_config.minimum_embedding_cluster_size:
                raise ValueError(
                    "embedding cluster is smaller than its configured minimum"
                )
            previous_parent = cluster_parent_identity.setdefault(
                metadata_group_id,
                relationship_identity,
            )
            if previous_parent != relationship_identity:
                raise ValueError(
                    "embedding clusters disagree about their metadata parent identity"
                )
            cluster_members = frozenset(
                zip(member_ids, member_fingerprints, strict=True)
            )
            previous_cluster = cluster_group_members.setdefault(
                group_id,
                (metadata_group_id, cluster_members),
            )
            if previous_cluster != (metadata_group_id, cluster_members):
                raise ValueError("embedding cluster membership is inconsistent")
            previous_policy = cluster_policy_by_parent.setdefault(
                metadata_group_id,
                clustering_config,
            )
            if previous_policy != clustering_config:
                raise ValueError(
                    "embedding clusters under one metadata parent use mixed policies"
                )
    for methods in group_methods.values():
        if PROTOTYPE_METHOD_NORMALIZED_MEAN not in methods:
            raise ValueError("reference prototype group lacks its raw centroid")
    for metadata_group_id, identity in cluster_parent_identity.items():
        if metadata_group_identity.get(metadata_group_id) != identity:
            raise ValueError(
                "embedding cluster metadata parent is missing or has another taxon identity"
            )
    clusters_by_parent: dict[str, list[frozenset[tuple[str, str]]]] = defaultdict(list)
    for metadata_group_id, members in cluster_group_members.values():
        clusters_by_parent[metadata_group_id].append(members)
    for metadata_group_id, cluster_memberships in clusters_by_parent.items():
        parent_members = metadata_group_members[metadata_group_id]
        policy = cluster_policy_by_parent[metadata_group_id]
        if not (
            policy.minimum_clustering_observation_count
            <= len(parent_members)
            <= policy.maximum_clustering_observation_count
        ):
            raise ValueError(
                "embedding cluster parent is outside its configured support bounds"
            )
        if not 2 <= len(cluster_memberships) <= policy.maximum_embedding_cluster_count:
            raise ValueError(
                "embedding cluster count violates its configured multi-prototype bounds"
            )
        combined: set[tuple[str, str]] = set()
        for members in cluster_memberships:
            if not members <= parent_members:
                raise ValueError(
                    "embedding cluster contains observations outside its metadata parent"
                )
            if combined.intersection(members):
                raise ValueError(
                    "embedding clusters duplicate observations within a metadata group"
                )
            combined.update(members)
        if combined != set(parent_members):
            raise ValueError(
                "embedding clusters do not cover their complete metadata group"
            )


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
    *,
    preserve_view: bool = False,
) -> tuple[ReferenceObservationEmbedding, ...]:
    group_fields = [
        "accepted_taxon_key",
        "scientific_name",
        "geo_cluster_id",
        "life_stage",
        "visual_domain",
    ]
    if preserve_view:
        group_fields.append("view")
    group_fields.extend(
        [
            "route",
            "visual_input_kind",
            "reference_observation_id",
        ]
    )
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    observation_identity: dict[tuple[str, ...], tuple[str, ...]] = {}
    for row in support.iter_rows(named=True):
        key = tuple(str(row[field]) for field in group_fields)
        identity_key = (
            str(row["reference_observation_id"]),
            str(row["visual_input_kind"]),
            str(row["view"]) if preserve_view else PROTOTYPE_AGGREGATE_VIEW,
        )
        semantic_identity = key[:-1]
        previous = observation_identity.setdefault(identity_key, semantic_identity)
        if previous != semantic_identity:
            raise ValueError(
                "reference observation crosses taxon, route, domain, or geographic identity"
            )
        grouped[key].append(row)

    observations: list[ReferenceObservationEmbedding] = []
    for key, rows in grouped.items():
        identity = dict(zip(group_fields, key, strict=True))
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
        duplicate_group_ids = tuple(
            sorted({str(row["duplicate_group_id"]) for row in rows})
        )
        payload = {
            "accepted_taxon_key": identity["accepted_taxon_key"],
            "scientific_name": identity["scientific_name"],
            "geo_cluster_id": identity["geo_cluster_id"],
            "life_stage": identity["life_stage"],
            "visual_domain": identity["visual_domain"],
            "view": (identity["view"] if preserve_view else PROTOTYPE_AGGREGATE_VIEW),
            "route": identity["route"],
            "visual_input_kind": identity["visual_input_kind"],
            "reference_observation_id": identity["reference_observation_id"],
            "contributor_fingerprints": contributor_fingerprints,
            "embedding": embedding,
        }
        observations.append(
            ReferenceObservationEmbedding(
                accepted_taxon_key=identity["accepted_taxon_key"],
                scientific_name=identity["scientific_name"],
                geo_cluster_id=identity["geo_cluster_id"],
                life_stage=identity["life_stage"],
                visual_domain=identity["visual_domain"],
                view=(identity["view"] if preserve_view else PROTOTYPE_AGGREGATE_VIEW),
                route=identity["route"],
                visual_input_kind=identity["visual_input_kind"],
                reference_observation_id=identity["reference_observation_id"],
                duplicate_group_ids=duplicate_group_ids,
                reference_count=len(rows),
                embedding=embedding,
                contributor_fingerprints=contributor_fingerprints,
                observation_fingerprint=canonical_semantic_fingerprint(payload),
            )
        )
    return tuple(sorted(observations, key=_observation_sort_key))


def _centering_contexts(
    observations: Sequence[ReferenceObservationEmbedding],
    *,
    embedding_dimension: int,
    balanced_sampling_seed: int,
) -> tuple[ReferenceCenteringContext, ...]:
    seed = _sampling_seed(balanced_sampling_seed)
    grouped: dict[tuple[str, str], list[ReferenceObservationEmbedding]] = defaultdict(
        list
    )
    for item in observations:
        grouped[(item.route, item.visual_input_kind)].append(item)
    contexts: list[ReferenceCenteringContext] = []
    for (route, visual_input_kind), items in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], _VISUAL_INPUT_KIND_ORDER[item[0][1]]),
    ):
        by_species: dict[str, list[ReferenceObservationEmbedding]] = defaultdict(list)
        for item in items:
            by_species[item.accepted_taxon_key].append(item)
        balanced_count = min(len(values) for values in by_species.values())
        selected: list[ReferenceObservationEmbedding] = []
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
    observations: Sequence[ReferenceObservationEmbedding],
) -> tuple[_PrototypeGroup, ...]:
    groups: dict[tuple[str, ...], list[ReferenceObservationEmbedding]] = defaultdict(
        list
    )
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


def _multi_prototype_groups(
    observations: Sequence[ReferenceObservationEmbedding],
    *,
    config: MultiPrototypeConfig,
) -> tuple[_PrototypeGroup, ...]:
    groups: dict[tuple[str, ...], list[ReferenceObservationEmbedding]] = defaultdict(
        list
    )
    for item in observations:
        base = (
            item.accepted_taxon_key,
            item.scientific_name,
            item.life_stage,
            item.visual_domain,
            item.route,
            item.visual_input_kind,
            item.view,
        )
        groups[(*base, PROTOTYPE_SCOPE_GLOBAL, PROTOTYPE_GLOBAL_GEO_CLUSTER_ID)].append(
            item
        )
        groups[(*base, PROTOTYPE_SCOPE_REGIONAL, item.geo_cluster_id)].append(item)

    result: list[_PrototypeGroup] = []
    for key, values in sorted(groups.items(), key=lambda item: item[0]):
        members = tuple(sorted(values, key=_observation_sort_key))
        metadata_group = _PrototypeGroup(
            cluster_scope_type=key[-2],
            geo_cluster_id=key[-1],
            observations=members,
            prototype_kind=PROTOTYPE_KIND_METADATA,
            view=key[-3],
        )
        clusters = _average_linkage_observation_clusters(members, config=config)
        if len(members) >= config.minimum_metadata_observation_count or clusters:
            result.append(metadata_group)
        if not clusters:
            continue
        metadata_group_id = _prototype_group_id(metadata_group)
        for cluster in clusters:
            result.append(
                _PrototypeGroup(
                    cluster_scope_type=metadata_group.cluster_scope_type,
                    geo_cluster_id=metadata_group.geo_cluster_id,
                    observations=cluster,
                    prototype_kind=PROTOTYPE_KIND_EMBEDDING_CLUSTER,
                    view=metadata_group.view,
                    metadata_group_id=metadata_group_id,
                    embedding_cluster_id=_embedding_cluster_id(
                        metadata_group_id=metadata_group_id,
                        observations=cluster,
                        config=config,
                    ),
                    clustering_config=config,
                )
            )
    return tuple(sorted(result, key=_prototype_group_sort_key))


def _average_linkage_observation_clusters(
    observations: Sequence[ReferenceObservationEmbedding],
    *,
    config: MultiPrototypeConfig,
) -> tuple[tuple[ReferenceObservationEmbedding, ...], ...]:
    if not config.enable_embedding_clustering:
        return ()
    ordered = tuple(sorted(observations, key=_observation_sort_key))
    count = len(ordered)
    if not (
        config.minimum_clustering_observation_count
        <= count
        <= config.maximum_clustering_observation_count
    ):
        return ()

    pairwise: dict[tuple[int, int], float] = {}
    for left in range(count):
        for right in range(left + 1, count):
            pairwise[(left, right)] = _cosine_distance(
                ordered[left].embedding,
                ordered[right].embedding,
            )

    clusters: dict[int, tuple[int, ...]] = {index: (index,) for index in range(count)}
    distances = dict(pairwise)
    next_cluster_id = count
    while len(clusters) > 1:
        best: tuple[float, tuple[str, ...], tuple[str, ...], int, int] | None = None
        cluster_ids = sorted(clusters)
        for position, left_id in enumerate(cluster_ids):
            for right_id in cluster_ids[position + 1 :]:
                key = _cluster_pair_key(left_id, right_id)
                candidate = (
                    distances[key],
                    _cluster_member_key(clusters[left_id], ordered),
                    _cluster_member_key(clusters[right_id], ordered),
                    left_id,
                    right_id,
                )
                if best is None or candidate < best:
                    best = candidate
        if best is None or best[0] > config.cosine_distance_threshold:
            break
        _, _, _, left_id, right_id = best
        left_members = clusters.pop(left_id)
        right_members = clusters.pop(right_id)
        merged_members = tuple(sorted((*left_members, *right_members)))
        for other_id in tuple(clusters):
            left_distance = distances[_cluster_pair_key(left_id, other_id)]
            right_distance = distances[_cluster_pair_key(right_id, other_id)]
            merged_distance = (
                len(left_members) * left_distance + len(right_members) * right_distance
            ) / len(merged_members)
            distances[_cluster_pair_key(next_cluster_id, other_id)] = merged_distance
        clusters[next_cluster_id] = merged_members
        next_cluster_id += 1

    minimum_size = config.minimum_embedding_cluster_size
    valid = [
        list(members) for members in clusters.values() if len(members) >= minimum_size
    ]
    small = [members for members in clusters.values() if len(members) < minimum_size]
    if len(valid) < 2:
        return ()
    valid.sort(key=lambda members: _cluster_member_key(members, ordered))
    for members in sorted(
        small,
        key=lambda value: _cluster_member_key(value, ordered),
    ):
        destination = min(
            range(len(valid)),
            key=lambda index: (
                _average_cluster_distance(members, valid[index], pairwise),
                _cluster_member_key(valid[index], ordered),
            ),
        )
        valid[destination].extend(members)
        valid[destination].sort()

    while len(valid) > config.maximum_embedding_cluster_count:
        best_pair = min(
            (
                (
                    _average_cluster_distance(valid[left], valid[right], pairwise),
                    _cluster_member_key(valid[left], ordered),
                    _cluster_member_key(valid[right], ordered),
                    left,
                    right,
                )
                for left in range(len(valid))
                for right in range(left + 1, len(valid))
            )
        )
        left = best_pair[-2]
        right = best_pair[-1]
        valid[left] = sorted((*valid[left], *valid[right]))
        del valid[right]

    result = tuple(
        tuple(ordered[index] for index in members)
        for members in sorted(
            valid,
            key=lambda value: _cluster_member_key(value, ordered),
        )
    )
    if len(result) < 2:
        return ()
    return result


def _average_cluster_distance(
    left: Sequence[int],
    right: Sequence[int],
    pairwise: Mapping[tuple[int, int], float],
) -> float:
    return fsum(
        pairwise[_cluster_pair_key(left_index, right_index)]
        for left_index in left
        for right_index in right
    ) / (len(left) * len(right))


def _cluster_pair_key(left: int, right: int) -> tuple[int, int]:
    if left == right:
        raise ValueError("cluster distance requires distinct clusters")
    return (left, right) if left < right else (right, left)


def _cluster_member_key(
    members: Sequence[int],
    observations: Sequence[ReferenceObservationEmbedding],
) -> tuple[str, ...]:
    return tuple(
        sorted(observations[index].reference_observation_id for index in members)
    )


def _cosine_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    similarity = fsum(a * b for a, b in zip(left, right, strict=True))
    return 1.0 - min(1.0, max(-1.0, similarity))


def _embedding_cluster_id(
    *,
    metadata_group_id: str,
    observations: Sequence[ReferenceObservationEmbedding],
    config: MultiPrototypeConfig,
) -> str:
    member_pairs = sorted(
        (
            item.reference_observation_id,
            item.observation_fingerprint,
        )
        for item in observations
    )
    return _embedding_cluster_id_from_members(
        metadata_group_id=metadata_group_id,
        member_ids=[item[0] for item in member_pairs],
        member_fingerprints=[item[1] for item in member_pairs],
        clustering_configuration_fingerprint=config.fingerprint,
    )


def _embedding_cluster_id_from_members(
    *,
    metadata_group_id: str,
    member_ids: Sequence[str],
    member_fingerprints: Sequence[str],
    clustering_configuration_fingerprint: str,
) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": "reference-embedding-cluster-v1",
            "metadata_group_id": metadata_group_id,
            "clustering_configuration_fingerprint": (
                clustering_configuration_fingerprint
            ),
            "member_observations": [
                {
                    "reference_observation_id": observation_id,
                    "observation_fingerprint": observation_fingerprint,
                }
                for observation_id, observation_fingerprint in zip(
                    member_ids,
                    member_fingerprints,
                    strict=True,
                )
            ],
        }
    )


def _prototype_group_sort_key(group: _PrototypeGroup) -> tuple[object, ...]:
    first = group.observations[0]
    return (
        first.route,
        first.accepted_taxon_key,
        _PROTOTYPE_SCOPE_ORDER[group.cluster_scope_type],
        group.geo_cluster_id,
        first.life_stage,
        first.visual_domain,
        _VISUAL_INPUT_KIND_ORDER[first.visual_input_kind],
        _PROTOTYPE_KIND_ORDER[group.prototype_kind],
        group.view,
        tuple(item.reference_observation_id for item in group.observations),
    )


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
            "view": group.view,
            "route": first.route,
            "visual_input_kind": first.visual_input_kind,
            "prototype_kind": group.prototype_kind,
            "metadata_group_id": group.metadata_group_id,
            "embedding_cluster_id": group.embedding_cluster_id,
            "clustering_configuration_fingerprint": (
                group.clustering_config.fingerprint
                if group.clustering_config is not None
                else None
            ),
            "observation_fingerprints": [
                item[1]
                for item in sorted(
                    (
                        observation.reference_observation_id,
                        observation.observation_fingerprint,
                    )
                    for observation in group.observations
                )
            ],
        }
    )


def _prototype_group_id_from_row(
    row: Mapping[str, object],
    *,
    member_fingerprints: Sequence[str],
    metadata_group_id: str | None,
    embedding_cluster_id: str | None,
    clustering_configuration_fingerprint: str | None,
) -> str:
    prototype_kind = str(row["prototype_kind"])
    semantic_metadata_group_id = (
        metadata_group_id
        if prototype_kind == PROTOTYPE_KIND_EMBEDDING_CLUSTER
        else None
    )
    return canonical_semantic_fingerprint(
        {
            "schema_version": REFERENCE_PROTOTYPE_GROUP_SCHEMA_VERSION,
            "accepted_taxon_key": row["accepted_taxon_key"],
            "species": row["species"],
            "cluster_scope_type": row["cluster_scope_type"],
            "geo_cluster_id": row["geo_cluster_id"],
            "life_stage": row["life_stage"],
            "visual_domain": row["visual_domain"],
            "view": row["view"],
            "route": row["route"],
            "visual_input_kind": row["visual_input_kind"],
            "prototype_kind": prototype_kind,
            "metadata_group_id": semantic_metadata_group_id,
            "embedding_cluster_id": embedding_cluster_id,
            "clustering_configuration_fingerprint": (
                clustering_configuration_fingerprint
            ),
            "observation_fingerprints": list(member_fingerprints),
        }
    )


def _validate_prototype_kind_fields(
    row: Mapping[str, object],
    *,
    prototype_kind: str,
    prototype_group_id: str,
    member_ids: Sequence[str],
    member_fingerprints: Sequence[str],
) -> tuple[
    str | None,
    str | None,
    str | None,
    MultiPrototypeConfig | None,
]:
    metadata_group_id = row["metadata_group_id"]
    embedding_cluster_id = row["embedding_cluster_id"]
    clustering_fields = (
        "clustering_method",
        "clustering_configuration_fingerprint",
        "clustering_cosine_distance_threshold",
        "clustering_minimum_metadata_observation_count",
        "clustering_minimum_observation_count",
        "clustering_minimum_cluster_size",
        "clustering_maximum_cluster_count",
        "clustering_maximum_observation_count",
    )
    if prototype_kind == PROTOTYPE_KIND_AGGREGATE:
        if row["view"] != PROTOTYPE_AGGREGATE_VIEW:
            raise ValueError("aggregate reference prototypes must use view=all")
        if metadata_group_id is not None or embedding_cluster_id is not None:
            raise ValueError("aggregate reference prototypes cannot declare group IDs")
        if any(row[field] is not None for field in clustering_fields):
            raise ValueError("aggregate reference prototypes cannot declare clustering")
        return None, None, None, None

    if row["view"] == PROTOTYPE_AGGREGATE_VIEW:
        raise ValueError("multi-prototypes must retain a concrete reference view")
    if prototype_kind == PROTOTYPE_KIND_METADATA:
        parsed_metadata_id = _sha256(
            metadata_group_id,
            field="metadata_group_id",
        )
        if parsed_metadata_id != prototype_group_id:
            raise ValueError("metadata prototype must identify its own group")
        if embedding_cluster_id is not None:
            raise ValueError("metadata prototype cannot declare an embedding cluster")
        if any(row[field] is not None for field in clustering_fields):
            raise ValueError("metadata prototype cannot declare clustering parameters")
        return parsed_metadata_id, None, None, None

    parsed_metadata_id = _sha256(metadata_group_id, field="metadata_group_id")
    parsed_cluster_id = _sha256(
        embedding_cluster_id,
        field="embedding_cluster_id",
    )
    if row["clustering_method"] != EMBEDDING_CLUSTER_METHOD_AVERAGE_LINKAGE_COSINE:
        raise ValueError("embedding prototype clustering method is unsupported")
    config = MultiPrototypeConfig(
        minimum_metadata_observation_count=_positive_integer(
            row["clustering_minimum_metadata_observation_count"],
            field="clustering_minimum_metadata_observation_count",
        ),
        enable_embedding_clustering=True,
        minimum_clustering_observation_count=_positive_integer(
            row["clustering_minimum_observation_count"],
            field="clustering_minimum_observation_count",
        ),
        minimum_embedding_cluster_size=_positive_integer(
            row["clustering_minimum_cluster_size"],
            field="clustering_minimum_cluster_size",
        ),
        maximum_embedding_cluster_count=_positive_integer(
            row["clustering_maximum_cluster_count"],
            field="clustering_maximum_cluster_count",
        ),
        maximum_clustering_observation_count=_positive_integer(
            row["clustering_maximum_observation_count"],
            field="clustering_maximum_observation_count",
        ),
        cosine_distance_threshold=_finite_float(
            row["clustering_cosine_distance_threshold"],
            field="clustering_cosine_distance_threshold",
        ),
    )
    clustering_fingerprint = _sha256(
        row["clustering_configuration_fingerprint"],
        field="clustering_configuration_fingerprint",
    )
    if clustering_fingerprint != config.fingerprint:
        raise ValueError("embedding prototype clustering configuration mismatch")
    expected_cluster_id = _embedding_cluster_id_from_members(
        metadata_group_id=parsed_metadata_id,
        member_ids=member_ids,
        member_fingerprints=member_fingerprints,
        clustering_configuration_fingerprint=clustering_fingerprint,
    )
    if parsed_cluster_id != expected_cluster_id:
        raise ValueError("embedding cluster ID is invalid")
    return parsed_metadata_id, parsed_cluster_id, clustering_fingerprint, config


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
    member_pairs = sorted(
        (
            item.reference_observation_id,
            item.observation_fingerprint,
        )
        for item in group.observations
    )
    metadata_group_id = (
        prototype_group_id
        if group.prototype_kind == PROTOTYPE_KIND_METADATA
        else group.metadata_group_id
    )
    clustering_config = group.clustering_config
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
        "view": group.view,
        "route": first.route,
        "visual_input_kind": first.visual_input_kind,
        "prototype_kind": group.prototype_kind,
        "prototype_method": prototype_method,
        "prototype_group_id": prototype_group_id,
        "metadata_group_id": metadata_group_id,
        "embedding_cluster_id": group.embedding_cluster_id,
        "clustering_method": (
            EMBEDDING_CLUSTER_METHOD_AVERAGE_LINKAGE_COSINE
            if clustering_config is not None
            else None
        ),
        "clustering_configuration_fingerprint": (
            clustering_config.fingerprint if clustering_config is not None else None
        ),
        "clustering_cosine_distance_threshold": (
            clustering_config.cosine_distance_threshold
            if clustering_config is not None
            else None
        ),
        "clustering_minimum_metadata_observation_count": (
            clustering_config.minimum_metadata_observation_count
            if clustering_config is not None
            else None
        ),
        "clustering_minimum_observation_count": (
            clustering_config.minimum_clustering_observation_count
            if clustering_config is not None
            else None
        ),
        "clustering_minimum_cluster_size": (
            clustering_config.minimum_embedding_cluster_size
            if clustering_config is not None
            else None
        ),
        "clustering_maximum_cluster_count": (
            clustering_config.maximum_embedding_cluster_count
            if clustering_config is not None
            else None
        ),
        "clustering_maximum_observation_count": (
            clustering_config.maximum_clustering_observation_count
            if clustering_config is not None
            else None
        ),
        "member_observation_ids": [item[0] for item in member_pairs],
        "member_observation_fingerprints": [item[1] for item in member_pairs],
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
            pl.col("prototype_kind")
            .replace_strict(_PROTOTYPE_KIND_ORDER, return_dtype=pl.UInt8)
            .alias("__kind_order"),
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
                "__kind_order",
                "view",
                "__method_order",
                "prototype_id",
            ]
        )
        .drop(
            "__scope_order",
            "__visual_input_order",
            "__kind_order",
            "__method_order",
        )
    )


def _balanced_observation_rank(
    item: ReferenceObservationEmbedding,
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


def _observation_sort_key(
    item: ReferenceObservationEmbedding,
) -> tuple[object, ...]:
    return (
        item.route,
        _VISUAL_INPUT_KIND_ORDER[item.visual_input_kind],
        item.accepted_taxon_key,
        item.geo_cluster_id,
        item.view,
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


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _sorted_unique_text_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty string list")
    result = tuple(_required_text(item, field=field) for item in value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{field} must be sorted and unique")
    return result


def _sha256_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty fingerprint list")
    result = tuple(_sha256(item, field=field) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique fingerprints")
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
    "EMBEDDING_CLUSTER_METHOD_AVERAGE_LINKAGE_COSINE",
    "MULTI_PROTOTYPE_CONFIG_SCHEMA_VERSION",
    "PROTOTYPE_AGGREGATE_VIEW",
    "PROTOTYPE_GLOBAL_GEO_CLUSTER_ID",
    "PROTOTYPE_KIND_AGGREGATE",
    "PROTOTYPE_KIND_EMBEDDING_CLUSTER",
    "PROTOTYPE_KIND_METADATA",
    "PROTOTYPE_METHOD_NORMALIZED_MEAN",
    "PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED",
    "PROTOTYPE_SCOPE_GLOBAL",
    "PROTOTYPE_SCOPE_REGIONAL",
    "REFERENCE_PROTOTYPES_FILE",
    "REFERENCE_PROTOTYPES_SCHEMA_VERSION",
    "REFERENCE_PROTOTYPE_METHODS",
    "REFERENCE_PROTOTYPE_KINDS",
    "REFERENCE_PROTOTYPE_SCOPE_TYPES",
    "ReferenceCenteringContext",
    "ReferenceObservationEmbedding",
    "MultiPrototypeConfig",
    "aggregate_reference_observation_embeddings",
    "build_reference_centering_contexts",
    "build_multi_reference_prototypes",
    "build_reference_prototypes",
    "load_reference_prototypes",
    "mean_center_query_embedding",
    "reference_prototypes_artifact_fingerprint",
    "reference_prototypes_schema",
    "validate_reference_prototypes",
    "write_reference_prototypes",
]
