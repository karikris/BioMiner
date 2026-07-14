"""Leakage-aware feature matrices over frozen BioCLIP embeddings."""

from __future__ import annotations

from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
import json
import logging
from pathlib import Path
import re

import polars as pl

from biominer.bioclip.reference_scoring import CandidateReferenceEvidence
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.detection.routing import DETECTION_ROUTES
from biominer.flickr_fetch.geographic_clustering import NO_GEO_CLUSTER_ID
from biominer.references.readiness import REFERENCE_ROUTES, REFERENCE_SUPPORT_SPLITS
from biominer.references.schemas import REFERENCE_VISUAL_DOMAINS
from biominer.storage.parquet import write_parquet
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
)


FEW_SHOT_TRAINING_FEATURES_SCHEMA_VERSION = "few-shot-training-features-v1.0.0"
FEW_SHOT_TRAINING_FEATURES_FILE = "few_shot_training_features.parquet"
FEATURE_SCHEMA_FINGERPRINT_VERSION = "few-shot-feature-schema-v1"
TRAINING_EXAMPLE_ID_VERSION = "few-shot-training-example-id-v1"
TRAINING_EXAMPLE_FINGERPRINT_VERSION = "few-shot-training-example-fingerprint-v1"
TRAINING_DATA_FINGERPRINT_VERSION = "few-shot-training-data-fingerprint-v1"
REFERENCE_FEATURE_DERIVATION_VERSION = "centroid-competitor-margin-v1"
TEXT_FEATURE_DERIVATION_VERSION = "text-similarity-margin-v1"
RESOLUTION_FEATURE_DERIVATION_VERSION = "short-side-224-v1"

LOW_RESOLUTION_SHORT_SIDE_PX = 224
TARGET_TASKS = frozenset(
    {
        "binary_target_verifier",
        "regional_multiclass",
        "visual_domain",
        "larval_target_verifier",
    }
)
LABEL_CERTAINTIES = frozenset({"high", "medium", "low", "unknown"})
_VISUAL_INPUT_KINDS = frozenset(
    {
        RAW_FULL_IMAGE_KIND,
        FOCUSED_FULL_FRAME_KIND,
        MASKED_FULL_FRAME_KIND,
        MULTI_OBJECT_FULL_FRAME_KIND,
    }
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UNIT_NORM_TOLERANCE = 1e-5
_FLOAT_EQUAL_TOLERANCE = 1e-12
_LOGGER = logging.getLogger(__name__)


NUMERIC_MODEL_FEATURE_COLUMNS = (
    "embedding",
    "target_reference_centroid_similarity",
    "target_nearest_reference_similarity",
    "target_top_three_mean_similarity",
    "target_top_five_mean_similarity",
    "target_local_prototype_similarity",
    "target_global_prototype_similarity",
    "best_regional_competitor_similarity",
    "best_same_genus_competitor_similarity",
    "best_false_positive_competitor_similarity",
    "best_family_negative_similarity",
    "best_domain_negative_similarity",
    "target_minus_best_competitor_margin",
    "target_minus_domain_negative_margin",
    "target_prototype_distance",
    "nearest_target_support_distance",
    "target_text_ensemble_similarity",
    "best_competitor_text_similarity",
    "target_minus_competitor_text_margin",
    "target_regional_overlap_score",
    "best_competitor_regional_overlap_score",
    "nearest_target_occurrence_cell_distance_km",
    "nearest_target_support_observation_distance_km",
    "target_candidate_source_count",
    "competitor_candidate_source_count",
    "total_candidate_source_count",
    "missing_geo",
    "detector_confidence",
    "subject_area_ratio",
    "mask_coverage",
    "multiple_organism_indicator",
    "image_width_px",
    "image_height_px",
    "image_short_side_px",
    "image_long_side_px",
    "image_megapixels",
    "low_resolution_indicator",
)
CATEGORICAL_MODEL_FEATURE_COLUMNS = (
    "route",
    "visual_input_kind",
    "yoloe_route",
    "visual_input_quality_flags",
)
MODEL_FEATURE_COLUMNS = (
    *NUMERIC_MODEL_FEATURE_COLUMNS,
    *CATEGORICAL_MODEL_FEATURE_COLUMNS,
)
LABEL_COLUMNS = (
    "target_present",
    "accepted_class_taxon_key",
    "visual_domain_label",
    "label_certainty",
    "species_training_suitable",
    "ambiguity_reason",
)
PROHIBITED_SOURCE_FEATURE_FIELDS = frozenset(
    {
        "discovery_taxon_key",
        "flickr_query_definition_id",
        "flickr_query_term",
        "label_source",
        "query_definition_id",
        "query_term",
        "search_term",
        "source_label",
        "source_query_term",
    }
)


@dataclass(frozen=True, slots=True)
class TrainingProvenance:
    source_item_id: str
    leakage_group_id: str
    geo_cluster_id: str
    dataset_split: str
    support_manifest_fingerprint: str
    reference_embedding_fingerprint: str
    reference_prototype_fingerprint: str
    model_fingerprint: str
    source_observation_id: str | None = None
    source_owner_id: str | None = None
    duplicate_group_id: str | None = None
    burst_group_id: str | None = None
    provider_mirror_group_id: str | None = None
    candidate_set_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingLabel:
    reviewed_label_id: str
    reviewed_label_fingerprint: str
    target_present: bool
    accepted_class_taxon_key: str | None
    visual_domain_label: str
    label_certainty: str
    species_training_suitable: bool
    ambiguity_reason: str | None = None


@dataclass(frozen=True, slots=True)
class FrozenEmbeddingFeatures:
    visual_input_id: str
    visual_input_kind: str
    embedding: tuple[float, ...]
    embedding_fingerprint: str
    model_fingerprint: str


@dataclass(frozen=True, slots=True)
class ReferenceEvidenceFeatures:
    model_fingerprint: str
    reference_embedding_fingerprint: str
    reference_prototype_fingerprint: str
    support_manifest_fingerprint: str
    route: str
    visual_input_kind: str
    geo_cluster_id: str
    target_centroid_similarity: float | None = None
    target_nearest_similarity: float | None = None
    target_top_three_mean_similarity: float | None = None
    target_top_five_mean_similarity: float | None = None
    target_local_prototype_similarity: float | None = None
    target_global_prototype_similarity: float | None = None
    regional_competitor_similarities: tuple[float, ...] = ()
    same_genus_competitor_similarities: tuple[float, ...] = ()
    false_positive_competitor_similarities: tuple[float, ...] = ()
    family_negative_similarities: tuple[float, ...] = ()
    domain_negative_similarities: tuple[float, ...] = ()

    @classmethod
    def from_candidate_evidence(
        cls,
        evidence: Sequence[CandidateReferenceEvidence],
        *,
        target_accepted_taxon_key: str,
        regional_competitor_taxon_keys: Sequence[str] = (),
        same_genus_competitor_taxon_keys: Sequence[str] = (),
        false_positive_competitor_taxon_keys: Sequence[str] = (),
        family_negative_taxon_keys: Sequence[str] = (),
        domain_negative_similarities: Sequence[float] = (),
    ) -> ReferenceEvidenceFeatures:
        """Adapt Task 7 scores using centroid similarity for class comparison."""

        target_key = _required_text(
            target_accepted_taxon_key,
            field="target_accepted_taxon_key",
        )
        items = tuple(evidence)
        if any(not isinstance(item, CandidateReferenceEvidence) for item in items):
            raise TypeError("evidence must contain CandidateReferenceEvidence values")
        by_key = {item.accepted_taxon_key: item for item in items}
        if len(by_key) != len(items):
            raise ValueError("candidate reference evidence contains duplicate taxa")
        target = by_key.get(target_key)
        if target is None:
            raise ValueError("candidate reference evidence does not contain the target")
        contract_fields = (
            "scoring_version",
            "query_id",
            "route",
            "visual_input_kind",
            "geo_cluster_id",
            "prototype_method",
            "balanced_sampling_seed",
            "fixed_reference_count",
            "model_fingerprint",
            "reference_embedding_fingerprint",
            "reference_prototype_fingerprint",
            "support_manifest_fingerprint",
        )
        contracts = {
            tuple(getattr(item, field) for field in contract_fields) for item in items
        }
        if len(contracts) != 1:
            raise ValueError("candidate reference evidence mixes scoring contracts")

        def category_scores(keys: Sequence[str], *, field: str) -> tuple[float, ...]:
            normalized = _unique_text_tuple(keys, field=field)
            if target_key in normalized:
                raise ValueError(f"{field} cannot contain the target taxon")
            unknown = sorted(set(normalized) - set(by_key))
            if unknown:
                raise ValueError(f"{field} contains taxa without evidence: {unknown}")
            return tuple(
                float(score)
                for key in normalized
                if key != target_key
                for score in (by_key[key].centroid_similarity,)
                if score is not None
            )

        return cls(
            model_fingerprint=target.model_fingerprint,
            reference_embedding_fingerprint=target.reference_embedding_fingerprint,
            reference_prototype_fingerprint=target.reference_prototype_fingerprint,
            support_manifest_fingerprint=target.support_manifest_fingerprint,
            route=target.route,
            visual_input_kind=target.visual_input_kind,
            geo_cluster_id=target.geo_cluster_id,
            target_centroid_similarity=target.centroid_similarity,
            target_nearest_similarity=target.nearest_support_similarity,
            target_top_three_mean_similarity=target.mean_top_three_similarity,
            target_top_five_mean_similarity=target.mean_top_five_similarity,
            target_local_prototype_similarity=(
                target.local_cluster_prototype_similarity
            ),
            target_global_prototype_similarity=target.global_prototype_similarity,
            regional_competitor_similarities=category_scores(
                regional_competitor_taxon_keys,
                field="regional_competitor_taxon_keys",
            ),
            same_genus_competitor_similarities=category_scores(
                same_genus_competitor_taxon_keys,
                field="same_genus_competitor_taxon_keys",
            ),
            false_positive_competitor_similarities=category_scores(
                false_positive_competitor_taxon_keys,
                field="false_positive_competitor_taxon_keys",
            ),
            family_negative_similarities=category_scores(
                family_negative_taxon_keys,
                field="family_negative_taxon_keys",
            ),
            domain_negative_similarities=tuple(domain_negative_similarities),
        )


@dataclass(frozen=True, slots=True)
class TextEvidenceFeatures:
    target_text_ensemble_similarity: float | None = None
    best_competitor_text_similarity: float | None = None


@dataclass(frozen=True, slots=True)
class GeographicEvidenceFeatures:
    target_regional_overlap_score: float | None = None
    best_competitor_regional_overlap_score: float | None = None
    nearest_target_occurrence_cell_distance_km: float | None = None
    nearest_target_support_observation_distance_km: float | None = None
    target_candidate_source_count: int = 0
    competitor_candidate_source_count: int = 0
    total_candidate_source_count: int = 0
    missing_geo: bool = False


@dataclass(frozen=True, slots=True)
class DetectionQualityFeatures:
    yoloe_route: str
    image_width_px: int
    image_height_px: int
    detector_confidence: float | None = None
    subject_area_ratio: float | None = None
    mask_coverage: float | None = None
    multiple_organism_indicator: bool = False
    visual_input_quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FewShotTrainingExample:
    target_task: str
    target_accepted_taxon_key: str
    route: str
    provenance: TrainingProvenance
    label: TrainingLabel
    embedding: FrozenEmbeddingFeatures
    reference: ReferenceEvidenceFeatures
    text: TextEvidenceFeatures
    geography: GeographicEvidenceFeatures
    detection: DetectionQualityFeatures


def few_shot_training_features_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "training_example_id": pl.String,
        "feature_schema_fingerprint": pl.String,
        "training_data_fingerprint": pl.String,
        "training_example_fingerprint": pl.String,
        "target_task": pl.String,
        "target_accepted_taxon_key": pl.String,
        "source_item_id": pl.String,
        "source_observation_id": pl.String,
        "source_owner_id": pl.String,
        "duplicate_group_id": pl.String,
        "burst_group_id": pl.String,
        "provider_mirror_group_id": pl.String,
        "leakage_group_id": pl.String,
        "geo_cluster_id": pl.String,
        "dataset_split": pl.String,
        "reviewed_label_id": pl.String,
        "reviewed_label_fingerprint": pl.String,
        "support_manifest_fingerprint": pl.String,
        "reference_embedding_fingerprint": pl.String,
        "reference_prototype_fingerprint": pl.String,
        "candidate_set_fingerprint": pl.String,
        "model_fingerprint": pl.String,
        "embedding_fingerprint": pl.String,
        "route": pl.String,
        "visual_input_id": pl.String,
        "visual_input_kind": pl.String,
        "embedding_dimension": pl.UInt32,
        "embedding": pl.List(pl.Float32),
        "embedding_norm": pl.Float64,
        "target_present": pl.Boolean,
        "accepted_class_taxon_key": pl.String,
        "visual_domain_label": pl.String,
        "label_certainty": pl.String,
        "species_training_suitable": pl.Boolean,
        "ambiguity_reason": pl.String,
        "target_reference_centroid_similarity": pl.Float64,
        "target_nearest_reference_similarity": pl.Float64,
        "target_top_three_mean_similarity": pl.Float64,
        "target_top_five_mean_similarity": pl.Float64,
        "target_local_prototype_similarity": pl.Float64,
        "target_global_prototype_similarity": pl.Float64,
        "best_regional_competitor_similarity": pl.Float64,
        "best_same_genus_competitor_similarity": pl.Float64,
        "best_false_positive_competitor_similarity": pl.Float64,
        "best_family_negative_similarity": pl.Float64,
        "best_domain_negative_similarity": pl.Float64,
        "target_minus_best_competitor_margin": pl.Float64,
        "target_minus_domain_negative_margin": pl.Float64,
        "target_prototype_distance": pl.Float64,
        "nearest_target_support_distance": pl.Float64,
        "target_text_ensemble_similarity": pl.Float64,
        "best_competitor_text_similarity": pl.Float64,
        "target_minus_competitor_text_margin": pl.Float64,
        "target_regional_overlap_score": pl.Float64,
        "best_competitor_regional_overlap_score": pl.Float64,
        "nearest_target_occurrence_cell_distance_km": pl.Float64,
        "nearest_target_support_observation_distance_km": pl.Float64,
        "target_candidate_source_count": pl.UInt32,
        "competitor_candidate_source_count": pl.UInt32,
        "total_candidate_source_count": pl.UInt32,
        "missing_geo": pl.Boolean,
        "yoloe_route": pl.String,
        "detector_confidence": pl.Float64,
        "subject_area_ratio": pl.Float64,
        "mask_coverage": pl.Float64,
        "multiple_organism_indicator": pl.Boolean,
        "image_width_px": pl.UInt32,
        "image_height_px": pl.UInt32,
        "image_short_side_px": pl.UInt32,
        "image_long_side_px": pl.UInt32,
        "image_megapixels": pl.Float64,
        "low_resolution_indicator": pl.Boolean,
        "visual_input_quality_flags": pl.List(pl.String),
    }


def feature_schema_fingerprint(embedding_dimension: int) -> str:
    dimension = _positive_integer(
        embedding_dimension,
        field="embedding_dimension",
    )
    schema = few_shot_training_features_schema()
    return canonical_semantic_fingerprint(
        {
            "schema_version": FEATURE_SCHEMA_FINGERPRINT_VERSION,
            "artifact_schema_version": FEW_SHOT_TRAINING_FEATURES_SCHEMA_VERSION,
            "embedding_dimension": dimension,
            "model_features": [
                {"name": name, "dtype": str(schema[name])}
                for name in MODEL_FEATURE_COLUMNS
            ],
            "numeric_model_features": list(NUMERIC_MODEL_FEATURE_COLUMNS),
            "categorical_model_features": list(CATEGORICAL_MODEL_FEATURE_COLUMNS),
            "reference_derivation": REFERENCE_FEATURE_DERIVATION_VERSION,
            "text_derivation": TEXT_FEATURE_DERIVATION_VERSION,
            "resolution_derivation": RESOLUTION_FEATURE_DERIVATION_VERSION,
            "prohibited_source_features": sorted(PROHIBITED_SOURCE_FEATURE_FIELDS),
        }
    )


def build_few_shot_training_features(
    examples: Sequence[FewShotTrainingExample],
) -> pl.DataFrame:
    """Build one immutable, model-ready row per reviewed visual input."""

    if isinstance(examples, (str, bytes, bytearray)):
        raise TypeError("examples must be a sequence of FewShotTrainingExample values")
    items = tuple(examples)
    if not items:
        raise ValueError("at least one training example is required")
    if any(not isinstance(item, FewShotTrainingExample) for item in items):
        raise TypeError("examples must contain FewShotTrainingExample values")
    dimensions = {len(item.embedding.embedding) for item in items}
    if len(dimensions) != 1:
        raise ValueError("training examples must use one embedding dimension")
    dimension = _positive_integer(next(iter(dimensions)), field="embedding_dimension")
    schema_fingerprint = feature_schema_fingerprint(dimension)
    rows = [
        _example_row(
            item,
            embedding_dimension=dimension,
            schema_fingerprint=schema_fingerprint,
        )
        for item in items
    ]
    rows.sort(key=_row_sort_key)
    if len({str(row["training_example_id"]) for row in rows}) != len(rows):
        raise ValueError("training examples contain duplicate semantic identities")
    data_fingerprint = _training_data_fingerprint(
        schema_fingerprint=schema_fingerprint,
        example_fingerprints=[str(row["training_example_fingerprint"]) for row in rows],
    )
    for row in rows:
        row["training_data_fingerprint"] = data_fingerprint
    result = pl.DataFrame(
        rows,
        schema=few_shot_training_features_schema(),
        orient="row",
        strict=True,
    )
    validate_few_shot_training_features(result)
    _log_event(
        "few_shot_training_features_built",
        row_count=result.height,
        embedding_dimension=dimension,
        task_count=result["target_task"].n_unique(),
        group_count=result["leakage_group_id"].n_unique(),
        split_count=result["dataset_split"].n_unique(),
        feature_schema_fingerprint=schema_fingerprint,
        training_data_fingerprint=data_fingerprint,
    )
    return result


def validate_few_shot_training_features(
    frame: pl.DataFrame,
    *,
    expected_training_data_fingerprint: str | None = None,
    expected_feature_schema_fingerprint: str | None = None,
    expected_model_fingerprint: str | None = None,
) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("training features must be a Polars DataFrame")
    if dict(frame.schema) != few_shot_training_features_schema():
        raise ValueError("training feature frame physical schema mismatch")
    if frame.is_empty():
        raise ValueError("training feature frame must not be empty")
    if not frame.equals(_sort_feature_frame(frame)):
        raise ValueError("training feature frame is not deterministically sorted")
    if frame["training_example_id"].n_unique() != frame.height:
        raise ValueError("training feature frame contains duplicate example IDs")
    if frame["training_example_fingerprint"].n_unique() != frame.height:
        raise ValueError("training feature frame contains duplicate row fingerprints")

    dimensions = frame["embedding_dimension"].unique().to_list()
    if len(dimensions) != 1:
        raise ValueError("training feature frame has mixed embedding dimensions")
    dimension = _positive_integer(dimensions[0], field="embedding_dimension")
    schema_fingerprint = _single_sha256(
        frame,
        "feature_schema_fingerprint",
        expected_feature_schema_fingerprint,
    )
    if schema_fingerprint != feature_schema_fingerprint(dimension):
        raise ValueError("training feature schema fingerprint is invalid")
    data_fingerprint = _single_sha256(
        frame,
        "training_data_fingerprint",
        expected_training_data_fingerprint,
    )
    _single_sha256(frame, "model_fingerprint", expected_model_fingerprint)
    for field in (
        "support_manifest_fingerprint",
        "reference_embedding_fingerprint",
        "reference_prototype_fingerprint",
    ):
        _single_sha256(frame, field)

    _validate_group_split_isolation(frame)
    example_fingerprints: list[str] = []
    for row in frame.iter_rows(named=True):
        _validate_feature_row(
            row,
            embedding_dimension=dimension,
            schema_fingerprint=schema_fingerprint,
        )
        expected_row_fingerprint = _training_example_fingerprint(row)
        if row["training_example_fingerprint"] != expected_row_fingerprint:
            raise ValueError("training example fingerprint is invalid")
        example_fingerprints.append(expected_row_fingerprint)
    expected_data_fingerprint = _training_data_fingerprint(
        schema_fingerprint=schema_fingerprint,
        example_fingerprints=example_fingerprints,
    )
    if data_fingerprint != expected_data_fingerprint:
        raise ValueError("training data fingerprint is invalid")


def write_few_shot_training_features(
    frame: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    validate_few_shot_training_features(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= FEW_SHOT_TRAINING_FEATURES_FILE
    written = write_parquet(frame, destination, overwrite=overwrite)
    loaded = pl.read_parquet(written)
    validate_few_shot_training_features(loaded)
    if not frame.equals(loaded):
        raise ValueError("training feature Parquet round-trip mismatch")
    _log_event(
        "few_shot_training_features_written",
        artifact_path=str(written),
        row_count=frame.height,
        byte_count=written.stat().st_size,
        training_data_fingerprint=str(frame["training_data_fingerprint"][0]),
    )
    return written


def load_few_shot_training_features(
    path: str | Path,
    *,
    expected_training_data_fingerprint: str | None = None,
    expected_feature_schema_fingerprint: str | None = None,
    expected_model_fingerprint: str | None = None,
) -> pl.DataFrame:
    source = Path(path)
    if source.is_dir():
        source /= FEW_SHOT_TRAINING_FEATURES_FILE
    frame = pl.read_parquet(source)
    validate_few_shot_training_features(
        frame,
        expected_training_data_fingerprint=expected_training_data_fingerprint,
        expected_feature_schema_fingerprint=expected_feature_schema_fingerprint,
        expected_model_fingerprint=expected_model_fingerprint,
    )
    return frame


def _example_row(
    example: FewShotTrainingExample,
    *,
    embedding_dimension: int,
    schema_fingerprint: str,
) -> dict[str, object]:
    target_task = _required_choice(
        example.target_task,
        field="target_task",
        allowed=TARGET_TASKS,
    )
    target_key = _required_text(
        example.target_accepted_taxon_key,
        field="target_accepted_taxon_key",
    )
    route = _required_choice(example.route, field="route", allowed=REFERENCE_ROUTES)
    provenance = example.provenance
    if not isinstance(provenance, TrainingProvenance):
        raise TypeError("provenance must be TrainingProvenance")
    label = example.label
    if not isinstance(label, TrainingLabel):
        raise TypeError("label must be TrainingLabel")
    frozen = example.embedding
    if not isinstance(frozen, FrozenEmbeddingFeatures):
        raise TypeError("embedding must be FrozenEmbeddingFeatures")
    if len(frozen.embedding) != embedding_dimension:
        raise ValueError("training example embedding dimension is inconsistent")
    embedding, embedding_norm = _stored_embedding(frozen.embedding)
    visual_input_kind = _required_choice(
        frozen.visual_input_kind,
        field="visual_input_kind",
        allowed=_VISUAL_INPUT_KINDS,
    )
    _validate_embedding_contract(frozen, provenance=provenance)
    _validate_reference_evidence_contract(
        example.reference,
        provenance=provenance,
        route=route,
        visual_input_kind=visual_input_kind,
    )
    reference_fields = _reference_feature_fields(example.reference)
    text_fields = _text_feature_fields(example.text)
    geography_fields = _geographic_feature_fields(
        example.geography,
        geo_cluster_id=provenance.geo_cluster_id,
    )
    detection_fields = _detection_feature_fields(example.detection)
    label_fields = _label_fields(
        label,
        target_task=target_task,
        target_accepted_taxon_key=target_key,
    )
    source_item_id = _required_text(
        provenance.source_item_id,
        field="source_item_id",
    )
    leakage_group_id = _required_text(
        provenance.leakage_group_id,
        field="leakage_group_id",
    )
    geo_cluster_id = _required_text(
        provenance.geo_cluster_id,
        field="geo_cluster_id",
    )
    dataset_split = _required_choice(
        provenance.dataset_split,
        field="dataset_split",
        allowed=REFERENCE_SUPPORT_SPLITS,
    )
    visual_input_id = _required_text(
        frozen.visual_input_id,
        field="visual_input_id",
    )
    reviewed_label_id = _required_text(
        label.reviewed_label_id,
        field="reviewed_label_id",
    )
    training_example_id = _training_example_id(
        target_task=target_task,
        target_accepted_taxon_key=target_key,
        source_item_id=source_item_id,
        visual_input_id=visual_input_id,
        route=route,
        reviewed_label_id=reviewed_label_id,
        feature_schema_fingerprint=schema_fingerprint,
    )
    row: dict[str, object] = {
        "schema_version": FEW_SHOT_TRAINING_FEATURES_SCHEMA_VERSION,
        "training_example_id": training_example_id,
        "feature_schema_fingerprint": schema_fingerprint,
        "training_data_fingerprint": "",
        "training_example_fingerprint": "",
        "target_task": target_task,
        "target_accepted_taxon_key": target_key,
        "source_item_id": source_item_id,
        "source_observation_id": _optional_text(
            provenance.source_observation_id,
            field="source_observation_id",
        ),
        "source_owner_id": _optional_text(
            provenance.source_owner_id,
            field="source_owner_id",
        ),
        "duplicate_group_id": _optional_text(
            provenance.duplicate_group_id,
            field="duplicate_group_id",
        ),
        "burst_group_id": _optional_text(
            provenance.burst_group_id,
            field="burst_group_id",
        ),
        "provider_mirror_group_id": _optional_text(
            provenance.provider_mirror_group_id,
            field="provider_mirror_group_id",
        ),
        "leakage_group_id": leakage_group_id,
        "geo_cluster_id": geo_cluster_id,
        "dataset_split": dataset_split,
        "reviewed_label_id": reviewed_label_id,
        "reviewed_label_fingerprint": _sha256(
            label.reviewed_label_fingerprint,
            field="reviewed_label_fingerprint",
        ),
        "support_manifest_fingerprint": _sha256(
            provenance.support_manifest_fingerprint,
            field="support_manifest_fingerprint",
        ),
        "reference_embedding_fingerprint": _sha256(
            provenance.reference_embedding_fingerprint,
            field="reference_embedding_fingerprint",
        ),
        "reference_prototype_fingerprint": _sha256(
            provenance.reference_prototype_fingerprint,
            field="reference_prototype_fingerprint",
        ),
        "candidate_set_fingerprint": _optional_sha256(
            provenance.candidate_set_fingerprint,
            field="candidate_set_fingerprint",
        ),
        "model_fingerprint": _sha256(
            provenance.model_fingerprint,
            field="model_fingerprint",
        ),
        "embedding_fingerprint": _sha256(
            frozen.embedding_fingerprint,
            field="embedding_fingerprint",
        ),
        "route": route,
        "visual_input_id": visual_input_id,
        "visual_input_kind": visual_input_kind,
        "embedding_dimension": embedding_dimension,
        "embedding": list(embedding),
        "embedding_norm": embedding_norm,
        **label_fields,
        **reference_fields,
        **text_fields,
        **geography_fields,
        **detection_fields,
    }
    if set(row) != set(few_shot_training_features_schema()):
        missing = sorted(set(few_shot_training_features_schema()) - set(row))
        extra = sorted(set(row) - set(few_shot_training_features_schema()))
        raise AssertionError(
            f"training feature row mismatch: missing={missing}, extra={extra}"
        )
    row["training_example_fingerprint"] = _training_example_fingerprint(row)
    return row


def _label_fields(
    label: TrainingLabel,
    *,
    target_task: str,
    target_accepted_taxon_key: str,
) -> dict[str, object]:
    if not isinstance(label.target_present, bool):
        raise TypeError("target_present must be boolean")
    accepted_class = _optional_text(
        label.accepted_class_taxon_key,
        field="accepted_class_taxon_key",
    )
    if label.target_present and accepted_class != target_accepted_taxon_key:
        raise ValueError("target-present labels must use the target accepted key")
    if not label.target_present and accepted_class == target_accepted_taxon_key:
        raise ValueError("target-absent labels cannot use the target accepted key")
    if target_task == "regional_multiclass" and accepted_class is None:
        raise ValueError("regional multiclass labels require an accepted class key")
    visual_domain = _required_choice(
        label.visual_domain_label,
        field="visual_domain_label",
        allowed=REFERENCE_VISUAL_DOMAINS,
    )
    certainty = _required_choice(
        label.label_certainty,
        field="label_certainty",
        allowed=LABEL_CERTAINTIES,
    )
    if not isinstance(label.species_training_suitable, bool):
        raise TypeError("species_training_suitable must be boolean")
    ambiguity = _optional_text(label.ambiguity_reason, field="ambiguity_reason")
    if not label.species_training_suitable and ambiguity is None:
        raise ValueError("unsuitable labels require an ambiguity reason")
    return {
        "target_present": label.target_present,
        "accepted_class_taxon_key": accepted_class,
        "visual_domain_label": visual_domain,
        "label_certainty": certainty,
        "species_training_suitable": label.species_training_suitable,
        "ambiguity_reason": ambiguity,
    }


def _reference_feature_fields(
    evidence: ReferenceEvidenceFeatures,
) -> dict[str, object]:
    if not isinstance(evidence, ReferenceEvidenceFeatures):
        raise TypeError("reference must be ReferenceEvidenceFeatures")
    target_centroid = _optional_similarity(
        evidence.target_centroid_similarity,
        field="target_centroid_similarity",
    )
    target_nearest = _optional_similarity(
        evidence.target_nearest_similarity,
        field="target_nearest_similarity",
    )
    target_top_three = _optional_similarity(
        evidence.target_top_three_mean_similarity,
        field="target_top_three_mean_similarity",
    )
    target_top_five = _optional_similarity(
        evidence.target_top_five_mean_similarity,
        field="target_top_five_mean_similarity",
    )
    target_local = _optional_similarity(
        evidence.target_local_prototype_similarity,
        field="target_local_prototype_similarity",
    )
    target_global = _optional_similarity(
        evidence.target_global_prototype_similarity,
        field="target_global_prototype_similarity",
    )
    regional = _similarity_sequence(
        evidence.regional_competitor_similarities,
        field="regional_competitor_similarities",
    )
    same_genus = _similarity_sequence(
        evidence.same_genus_competitor_similarities,
        field="same_genus_competitor_similarities",
    )
    false_positive = _similarity_sequence(
        evidence.false_positive_competitor_similarities,
        field="false_positive_competitor_similarities",
    )
    family_negative = _similarity_sequence(
        evidence.family_negative_similarities,
        field="family_negative_similarities",
    )
    domain_negative = _similarity_sequence(
        evidence.domain_negative_similarities,
        field="domain_negative_similarities",
    )
    best_regional = _optional_max(regional)
    best_same_genus = _optional_max(same_genus)
    best_false_positive = _optional_max(false_positive)
    best_family_negative = _optional_max(family_negative)
    best_domain_negative = _optional_max(domain_negative)
    best_competitor = _optional_max(
        tuple(
            value
            for value in (
                best_regional,
                best_same_genus,
                best_false_positive,
                best_family_negative,
            )
            if value is not None
        )
    )
    return {
        "target_reference_centroid_similarity": target_centroid,
        "target_nearest_reference_similarity": target_nearest,
        "target_top_three_mean_similarity": target_top_three,
        "target_top_five_mean_similarity": target_top_five,
        "target_local_prototype_similarity": target_local,
        "target_global_prototype_similarity": target_global,
        "best_regional_competitor_similarity": best_regional,
        "best_same_genus_competitor_similarity": best_same_genus,
        "best_false_positive_competitor_similarity": best_false_positive,
        "best_family_negative_similarity": best_family_negative,
        "best_domain_negative_similarity": best_domain_negative,
        "target_minus_best_competitor_margin": _difference(
            target_centroid,
            best_competitor,
        ),
        "target_minus_domain_negative_margin": _difference(
            target_centroid,
            best_domain_negative,
        ),
        "target_prototype_distance": (
            1.0 - target_global if target_global is not None else None
        ),
        "nearest_target_support_distance": (
            1.0 - target_nearest if target_nearest is not None else None
        ),
    }


def _validate_embedding_contract(
    embedding: FrozenEmbeddingFeatures,
    *,
    provenance: TrainingProvenance,
) -> None:
    model_fingerprint = _sha256(
        embedding.model_fingerprint,
        field="embedding model_fingerprint",
    )
    expected = _sha256(
        provenance.model_fingerprint,
        field="provenance model_fingerprint",
    )
    if model_fingerprint != expected:
        raise ValueError("frozen embedding model fingerprint conflicts with provenance")


def _validate_reference_evidence_contract(
    evidence: ReferenceEvidenceFeatures,
    *,
    provenance: TrainingProvenance,
    route: str,
    visual_input_kind: str,
) -> None:
    if not isinstance(evidence, ReferenceEvidenceFeatures):
        raise TypeError("reference must be ReferenceEvidenceFeatures")
    comparisons = (
        (
            _sha256(evidence.model_fingerprint, field="reference model_fingerprint"),
            _sha256(provenance.model_fingerprint, field="model_fingerprint"),
            "model fingerprint",
        ),
        (
            _sha256(
                evidence.reference_embedding_fingerprint,
                field="evidence reference_embedding_fingerprint",
            ),
            _sha256(
                provenance.reference_embedding_fingerprint,
                field="reference_embedding_fingerprint",
            ),
            "reference embedding fingerprint",
        ),
        (
            _sha256(
                evidence.reference_prototype_fingerprint,
                field="evidence reference_prototype_fingerprint",
            ),
            _sha256(
                provenance.reference_prototype_fingerprint,
                field="reference_prototype_fingerprint",
            ),
            "reference prototype fingerprint",
        ),
        (
            _sha256(
                evidence.support_manifest_fingerprint,
                field="evidence support_manifest_fingerprint",
            ),
            _sha256(
                provenance.support_manifest_fingerprint,
                field="support_manifest_fingerprint",
            ),
            "support manifest fingerprint",
        ),
        (
            _required_choice(
                evidence.route,
                field="reference evidence route",
                allowed=REFERENCE_ROUTES,
            ),
            route,
            "route",
        ),
        (
            _required_choice(
                evidence.visual_input_kind,
                field="reference evidence visual_input_kind",
                allowed=_VISUAL_INPUT_KINDS,
            ),
            visual_input_kind,
            "visual-input kind",
        ),
        (
            _required_text(
                evidence.geo_cluster_id,
                field="reference evidence geo_cluster_id",
            ),
            _required_text(provenance.geo_cluster_id, field="geo_cluster_id"),
            "geo cluster",
        ),
    )
    for actual, expected, field in comparisons:
        if actual != expected:
            raise ValueError(f"reference evidence {field} conflicts with provenance")


def _text_feature_fields(evidence: TextEvidenceFeatures) -> dict[str, object]:
    if not isinstance(evidence, TextEvidenceFeatures):
        raise TypeError("text must be TextEvidenceFeatures")
    target = _optional_similarity(
        evidence.target_text_ensemble_similarity,
        field="target_text_ensemble_similarity",
    )
    competitor = _optional_similarity(
        evidence.best_competitor_text_similarity,
        field="best_competitor_text_similarity",
    )
    return {
        "target_text_ensemble_similarity": target,
        "best_competitor_text_similarity": competitor,
        "target_minus_competitor_text_margin": _difference(target, competitor),
    }


def _geographic_feature_fields(
    evidence: GeographicEvidenceFeatures,
    *,
    geo_cluster_id: str,
) -> dict[str, object]:
    if not isinstance(evidence, GeographicEvidenceFeatures):
        raise TypeError("geography must be GeographicEvidenceFeatures")
    if not isinstance(evidence.missing_geo, bool):
        raise TypeError("missing_geo must be boolean")
    cluster = _required_text(geo_cluster_id, field="geo_cluster_id")
    if evidence.missing_geo != (cluster == NO_GEO_CLUSTER_ID):
        raise ValueError("missing_geo must agree with geo_cluster_id")
    target_overlap = _optional_unit_interval(
        evidence.target_regional_overlap_score,
        field="target_regional_overlap_score",
    )
    competitor_overlap = _optional_unit_interval(
        evidence.best_competitor_regional_overlap_score,
        field="best_competitor_regional_overlap_score",
    )
    occurrence_distance = _optional_nonnegative_float(
        evidence.nearest_target_occurrence_cell_distance_km,
        field="nearest_target_occurrence_cell_distance_km",
    )
    support_distance = _optional_nonnegative_float(
        evidence.nearest_target_support_observation_distance_km,
        field="nearest_target_support_observation_distance_km",
    )
    if evidence.missing_geo and any(
        value is not None
        for value in (
            target_overlap,
            competitor_overlap,
            occurrence_distance,
            support_distance,
        )
    ):
        raise ValueError(
            "missing-geo examples cannot contain regional distances or overlap"
        )
    target_sources = _nonnegative_integer(
        evidence.target_candidate_source_count,
        field="target_candidate_source_count",
    )
    competitor_sources = _nonnegative_integer(
        evidence.competitor_candidate_source_count,
        field="competitor_candidate_source_count",
    )
    total_sources = _nonnegative_integer(
        evidence.total_candidate_source_count,
        field="total_candidate_source_count",
    )
    if total_sources < max(target_sources, competitor_sources):
        raise ValueError("total candidate-source count cannot be smaller than a subset")
    return {
        "target_regional_overlap_score": target_overlap,
        "best_competitor_regional_overlap_score": competitor_overlap,
        "nearest_target_occurrence_cell_distance_km": occurrence_distance,
        "nearest_target_support_observation_distance_km": support_distance,
        "target_candidate_source_count": target_sources,
        "competitor_candidate_source_count": competitor_sources,
        "total_candidate_source_count": total_sources,
        "missing_geo": evidence.missing_geo,
    }


def _detection_feature_fields(
    evidence: DetectionQualityFeatures,
) -> dict[str, object]:
    if not isinstance(evidence, DetectionQualityFeatures):
        raise TypeError("detection must be DetectionQualityFeatures")
    route = _required_choice(
        evidence.yoloe_route,
        field="yoloe_route",
        allowed=DETECTION_ROUTES,
    )
    width = _positive_integer(evidence.image_width_px, field="image_width_px")
    height = _positive_integer(evidence.image_height_px, field="image_height_px")
    short_side = min(width, height)
    long_side = max(width, height)
    if not isinstance(evidence.multiple_organism_indicator, bool):
        raise TypeError("multiple_organism_indicator must be boolean")
    return {
        "yoloe_route": route,
        "detector_confidence": _optional_unit_interval(
            evidence.detector_confidence,
            field="detector_confidence",
        ),
        "subject_area_ratio": _optional_unit_interval(
            evidence.subject_area_ratio,
            field="subject_area_ratio",
        ),
        "mask_coverage": _optional_unit_interval(
            evidence.mask_coverage,
            field="mask_coverage",
        ),
        "multiple_organism_indicator": evidence.multiple_organism_indicator,
        "image_width_px": width,
        "image_height_px": height,
        "image_short_side_px": short_side,
        "image_long_side_px": long_side,
        "image_megapixels": width * height / 1_000_000.0,
        "low_resolution_indicator": short_side < LOW_RESOLUTION_SHORT_SIDE_PX,
        "visual_input_quality_flags": list(
            _unique_text_tuple(
                evidence.visual_input_quality_flags,
                field="visual_input_quality_flags",
            )
        ),
    }


def _validate_feature_row(
    row: Mapping[str, object],
    *,
    embedding_dimension: int,
    schema_fingerprint: str,
) -> None:
    if row["schema_version"] != FEW_SHOT_TRAINING_FEATURES_SCHEMA_VERSION:
        raise ValueError("unsupported training feature schema version")
    for field in (
        "training_example_id",
        "target_task",
        "target_accepted_taxon_key",
        "source_item_id",
        "leakage_group_id",
        "geo_cluster_id",
        "dataset_split",
        "reviewed_label_id",
        "route",
        "visual_input_id",
        "visual_input_kind",
        "visual_domain_label",
        "label_certainty",
        "yoloe_route",
    ):
        _required_text(row[field], field=field)
    _required_choice(row["target_task"], field="target_task", allowed=TARGET_TASKS)
    _required_choice(row["route"], field="route", allowed=REFERENCE_ROUTES)
    _required_choice(
        row["visual_input_kind"],
        field="visual_input_kind",
        allowed=_VISUAL_INPUT_KINDS,
    )
    _required_choice(
        row["dataset_split"],
        field="dataset_split",
        allowed=REFERENCE_SUPPORT_SPLITS,
    )
    _required_choice(
        row["visual_domain_label"],
        field="visual_domain_label",
        allowed=REFERENCE_VISUAL_DOMAINS,
    )
    _required_choice(
        row["label_certainty"],
        field="label_certainty",
        allowed=LABEL_CERTAINTIES,
    )
    _required_choice(row["yoloe_route"], field="yoloe_route", allowed=DETECTION_ROUTES)
    if row["feature_schema_fingerprint"] != schema_fingerprint:
        raise ValueError("training row has conflicting feature schema fingerprint")
    for field in (
        "reviewed_label_fingerprint",
        "support_manifest_fingerprint",
        "reference_embedding_fingerprint",
        "reference_prototype_fingerprint",
        "model_fingerprint",
        "embedding_fingerprint",
        "training_example_fingerprint",
        "training_data_fingerprint",
    ):
        _sha256(row[field], field=field)
    _optional_sha256(
        row["candidate_set_fingerprint"], field="candidate_set_fingerprint"
    )
    vector, actual_norm = _stored_embedding(row["embedding"])
    if len(vector) != embedding_dimension:
        raise ValueError("training row embedding dimension is invalid")
    if int(row["embedding_dimension"]) != embedding_dimension:
        raise ValueError("training row embedding_dimension is invalid")
    stored_norm = _finite_float(row["embedding_norm"], field="embedding_norm")
    if abs(stored_norm - actual_norm) > _UNIT_NORM_TOLERANCE:
        raise ValueError("training row embedding_norm does not match embedding")
    for field in (
        "target_reference_centroid_similarity",
        "target_nearest_reference_similarity",
        "target_top_three_mean_similarity",
        "target_top_five_mean_similarity",
        "target_local_prototype_similarity",
        "target_global_prototype_similarity",
        "best_regional_competitor_similarity",
        "best_same_genus_competitor_similarity",
        "best_false_positive_competitor_similarity",
        "best_family_negative_similarity",
        "best_domain_negative_similarity",
        "target_text_ensemble_similarity",
        "best_competitor_text_similarity",
    ):
        _optional_similarity(row[field], field=field)
    for field in (
        "target_minus_best_competitor_margin",
        "target_minus_domain_negative_margin",
        "target_minus_competitor_text_margin",
    ):
        value = _optional_finite_float(row[field], field=field)
        if value is not None and not -2.0 <= value <= 2.0:
            raise ValueError(f"{field} must be in [-2, 2]")
    for field in ("target_prototype_distance", "nearest_target_support_distance"):
        value = _optional_finite_float(row[field], field=field)
        if value is not None and not 0.0 <= value <= 2.0:
            raise ValueError(f"{field} must be in [0, 2]")
    for field in (
        "target_regional_overlap_score",
        "best_competitor_regional_overlap_score",
        "detector_confidence",
        "subject_area_ratio",
        "mask_coverage",
    ):
        _optional_unit_interval(row[field], field=field)
    for field in (
        "nearest_target_occurrence_cell_distance_km",
        "nearest_target_support_observation_distance_km",
    ):
        _optional_nonnegative_float(row[field], field=field)
    target_sources = _nonnegative_integer(
        row["target_candidate_source_count"],
        field="target_candidate_source_count",
    )
    competitor_sources = _nonnegative_integer(
        row["competitor_candidate_source_count"],
        field="competitor_candidate_source_count",
    )
    total_sources = _nonnegative_integer(
        row["total_candidate_source_count"],
        field="total_candidate_source_count",
    )
    if total_sources < max(target_sources, competitor_sources):
        raise ValueError("training row candidate-source counts are inconsistent")
    for field in (
        "target_present",
        "species_training_suitable",
        "missing_geo",
        "multiple_organism_indicator",
        "low_resolution_indicator",
    ):
        if not isinstance(row[field], bool):
            raise ValueError(f"{field} must be boolean")
    target_key = str(row["target_accepted_taxon_key"])
    accepted_class = _optional_text(
        row["accepted_class_taxon_key"],
        field="accepted_class_taxon_key",
    )
    if bool(row["target_present"]) != (accepted_class == target_key):
        raise ValueError("training row target label is inconsistent")
    if row["target_task"] == "regional_multiclass" and accepted_class is None:
        raise ValueError("regional multiclass training row lacks an accepted class")
    for field in (
        "source_observation_id",
        "source_owner_id",
        "duplicate_group_id",
        "burst_group_id",
        "provider_mirror_group_id",
    ):
        _optional_text(row[field], field=field)
    ambiguity = _optional_text(row["ambiguity_reason"], field="ambiguity_reason")
    if not bool(row["species_training_suitable"]) and ambiguity is None:
        raise ValueError("unsuitable training row lacks an ambiguity reason")
    missing_geo = bool(row["missing_geo"])
    if missing_geo != (str(row["geo_cluster_id"]) == NO_GEO_CLUSTER_ID):
        raise ValueError("training row missing_geo is inconsistent")
    if missing_geo and any(
        row[field] is not None
        for field in (
            "target_regional_overlap_score",
            "best_competitor_regional_overlap_score",
            "nearest_target_occurrence_cell_distance_km",
            "nearest_target_support_observation_distance_km",
        )
    ):
        raise ValueError("missing-geo training row contains geographic evidence")
    width = _positive_integer(row["image_width_px"], field="image_width_px")
    height = _positive_integer(row["image_height_px"], field="image_height_px")
    if int(row["image_short_side_px"]) != min(width, height):
        raise ValueError("image_short_side_px is invalid")
    if int(row["image_long_side_px"]) != max(width, height):
        raise ValueError("image_long_side_px is invalid")
    if not _float_equal(float(row["image_megapixels"]), width * height / 1_000_000):
        raise ValueError("image_megapixels is invalid")
    if bool(row["low_resolution_indicator"]) != (
        min(width, height) < LOW_RESOLUTION_SHORT_SIDE_PX
    ):
        raise ValueError("low_resolution_indicator is invalid")
    quality_flags = row["visual_input_quality_flags"]
    if not isinstance(quality_flags, list):
        raise ValueError("visual_input_quality_flags must be a list")
    if quality_flags != list(
        _unique_text_tuple(quality_flags, field="visual_input_quality_flags")
    ):
        raise ValueError("visual_input_quality_flags must be sorted and unique")
    _validate_derived_features(row)
    expected_id = _training_example_id(
        target_task=str(row["target_task"]),
        target_accepted_taxon_key=target_key,
        source_item_id=str(row["source_item_id"]),
        visual_input_id=str(row["visual_input_id"]),
        route=str(row["route"]),
        reviewed_label_id=str(row["reviewed_label_id"]),
        feature_schema_fingerprint=schema_fingerprint,
    )
    if row["training_example_id"] != expected_id:
        raise ValueError("training example ID is invalid")


def _validate_derived_features(row: Mapping[str, object]) -> None:
    best_competitor = _optional_max(
        tuple(
            float(value)
            for value in (
                row["best_regional_competitor_similarity"],
                row["best_same_genus_competitor_similarity"],
                row["best_false_positive_competitor_similarity"],
                row["best_family_negative_similarity"],
            )
            if value is not None
        )
    )
    checks = {
        "target_minus_best_competitor_margin": _difference(
            _optional_similarity(
                row["target_reference_centroid_similarity"],
                field="target_reference_centroid_similarity",
            ),
            best_competitor,
        ),
        "target_minus_domain_negative_margin": _difference(
            _optional_similarity(
                row["target_reference_centroid_similarity"],
                field="target_reference_centroid_similarity",
            ),
            _optional_similarity(
                row["best_domain_negative_similarity"],
                field="best_domain_negative_similarity",
            ),
        ),
        "target_minus_competitor_text_margin": _difference(
            _optional_similarity(
                row["target_text_ensemble_similarity"],
                field="target_text_ensemble_similarity",
            ),
            _optional_similarity(
                row["best_competitor_text_similarity"],
                field="best_competitor_text_similarity",
            ),
        ),
        "target_prototype_distance": (
            1.0 - float(row["target_global_prototype_similarity"])
            if row["target_global_prototype_similarity"] is not None
            else None
        ),
        "nearest_target_support_distance": (
            1.0 - float(row["target_nearest_reference_similarity"])
            if row["target_nearest_reference_similarity"] is not None
            else None
        ),
    }
    for field, expected in checks.items():
        actual = _optional_finite_float(row[field], field=field)
        if not _optional_float_equal(actual, expected):
            raise ValueError(f"{field} is inconsistent with source similarities")


def _validate_group_split_isolation(frame: pl.DataFrame) -> None:
    for field in (
        "leakage_group_id",
        "source_observation_id",
        "source_owner_id",
        "duplicate_group_id",
        "burst_group_id",
        "provider_mirror_group_id",
        "geo_cluster_id",
    ):
        assignments: dict[str, str] = {}
        for value, split in frame.select(field, "dataset_split").iter_rows():
            if value is None or (
                field == "geo_cluster_id" and value == NO_GEO_CLUSTER_ID
            ):
                continue
            key = str(value)
            previous = assignments.setdefault(key, str(split))
            if previous != str(split):
                raise ValueError(
                    f"{field} {key!r} crosses dataset splits: "
                    f"{previous!r} and {split!r}"
                )


def _training_example_id(
    *,
    target_task: str,
    target_accepted_taxon_key: str,
    source_item_id: str,
    visual_input_id: str,
    route: str,
    reviewed_label_id: str,
    feature_schema_fingerprint: str,
) -> str:
    digest = canonical_semantic_fingerprint(
        {
            "schema_version": TRAINING_EXAMPLE_ID_VERSION,
            "target_task": target_task,
            "target_accepted_taxon_key": target_accepted_taxon_key,
            "source_item_id": source_item_id,
            "visual_input_id": visual_input_id,
            "route": route,
            "reviewed_label_id": reviewed_label_id,
            "feature_schema_fingerprint": feature_schema_fingerprint,
        }
    ).removeprefix("sha256:")
    return f"few-shot-training-example:{digest}"


def _training_example_fingerprint(row: Mapping[str, object]) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": TRAINING_EXAMPLE_FINGERPRINT_VERSION,
            "row": {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "training_data_fingerprint",
                    "training_example_fingerprint",
                }
            },
        }
    )


def _training_data_fingerprint(
    *,
    schema_fingerprint: str,
    example_fingerprints: Sequence[str],
) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": TRAINING_DATA_FINGERPRINT_VERSION,
            "feature_schema_fingerprint": schema_fingerprint,
            "training_example_fingerprints": list(example_fingerprints),
        }
    )


def _row_sort_key(row: Mapping[str, object]) -> tuple[str, ...]:
    return (
        str(row["dataset_split"]),
        str(row["leakage_group_id"]),
        str(row["source_item_id"]),
        str(row["visual_input_id"]),
        str(row["route"]),
        str(row["target_task"]),
        str(row["training_example_id"]),
    )


def _sort_feature_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort(
        [
            "dataset_split",
            "leakage_group_id",
            "source_item_id",
            "visual_input_id",
            "route",
            "target_task",
            "training_example_id",
        ]
    )


def _stored_embedding(values: object) -> tuple[tuple[float, ...], float]:
    if not isinstance(values, Sequence) or isinstance(
        values,
        (str, bytes, bytearray),
    ):
        raise ValueError("embedding must be a numeric sequence")
    try:
        stored = tuple(float(value) for value in array("f", values))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("embedding must contain finite Float32 values") from exc
    if not stored or any(not isfinite(value) for value in stored):
        raise ValueError("embedding must contain finite Float32 values")
    norm = sqrt(fsum(value * value for value in stored))
    if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
        raise ValueError("frozen embedding must be unit-normalized")
    return stored, norm


def _similarity_sequence(values: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(
        values,
        (str, bytes, bytearray),
    ):
        raise TypeError(f"{field} must be a sequence")
    return tuple(_similarity(value, field=field) for value in values)


def _optional_max(values: Sequence[float]) -> float | None:
    return max(values) if values else None


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _optional_similarity(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _similarity(value, field=field)


def _similarity(value: object, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [-1, 1]")
    return result


def _optional_unit_interval(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    result = _finite_float(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def _optional_nonnegative_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    result = _finite_float(value, field=field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _optional_finite_float(value: object, *, field: str) -> float | None:
    return None if value is None else _finite_float(value, field=field)


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


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _required_choice(
    value: object,
    *,
    field: str,
    allowed: frozenset[str] | Sequence[str],
) -> str:
    result = _required_text(value, field=field)
    if result not in allowed:
        raise ValueError(f"unsupported {field}: {result}")
    return result


def _unique_text_tuple(values: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(
        values,
        (str, bytes, bytearray),
    ):
        raise TypeError(f"{field} must be a sequence")
    normalized = tuple(_required_text(value, field=field) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must not contain duplicates")
    return tuple(sorted(normalized))


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 fingerprint")
    return value


def _optional_sha256(value: object, *, field: str) -> str | None:
    return None if value is None else _sha256(value, field=field)


def _single_sha256(
    frame: pl.DataFrame,
    field: str,
    expected: str | None = None,
) -> str:
    values = frame[field].unique().to_list()
    if len(values) != 1:
        raise ValueError(f"training feature frame has mixed {field} values")
    value = _sha256(values[0], field=field)
    if expected is not None and value != _sha256(expected, field=f"expected {field}"):
        raise ValueError(f"training feature {field} does not match expected")
    return value


def _float_equal(left: float, right: float) -> bool:
    return abs(left - right) <= _FLOAT_EQUAL_TOLERANCE


def _optional_float_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return _float_equal(left, right)


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
    "CATEGORICAL_MODEL_FEATURE_COLUMNS",
    "FEW_SHOT_TRAINING_FEATURES_FILE",
    "FEW_SHOT_TRAINING_FEATURES_SCHEMA_VERSION",
    "LABEL_COLUMNS",
    "MODEL_FEATURE_COLUMNS",
    "NUMERIC_MODEL_FEATURE_COLUMNS",
    "PROHIBITED_SOURCE_FEATURE_FIELDS",
    "DetectionQualityFeatures",
    "FewShotTrainingExample",
    "FrozenEmbeddingFeatures",
    "GeographicEvidenceFeatures",
    "ReferenceEvidenceFeatures",
    "TextEvidenceFeatures",
    "TrainingLabel",
    "TrainingProvenance",
    "build_few_shot_training_features",
    "feature_schema_fingerprint",
    "few_shot_training_features_schema",
    "load_few_shot_training_features",
    "validate_few_shot_training_features",
    "write_few_shot_training_features",
]
