from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
from pathlib import Path
from shutil import rmtree
import tempfile
from typing import Any, Iterable, Iterator, Literal, Mapping, Protocol, Sequence

import polars as pl

from biominer.bioclip.cascade_contract import (
    DEFAULT_SPECIES_REPORT_TOP_K,
    GLOBAL_RANK_TOP_K_BEAM_STRATEGY,
    validate_production_cascade_settings,
)
from biominer.bioclip.candidate_sets import CandidateSet, CandidateTaxon
from biominer.bioclip.classification_modes import (
    DEFAULT_CLASSIFICATION_MODE,
    DEFAULT_FAMILY_TOP_K,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    ClassificationMode,
    normalize_classification_mode,
)
from biominer.bioclip.five_rank_classifier import (
    FIVE_RANK_OBJECT_SCORE_SCHEMA_EXTENSIONS,
)
from biominer.bioclip.path_cascade_classifier import classify_path_cascade_batch
from biominer.bioclip.path_cascade_output import (
    PATH_CASCADE_OUTPUT_SCHEMA,
    path_cascade_result_to_object_score_row,
)
from biominer.bioclip.path_taxonomy_store import PathTaxonomyStore
from biominer.bioclip.policy import DEFAULT_BUCKET_POLICY
from biominer.bioclip.taxonomy_embedding_cache import TaxonomyTextEmbeddingIndex
from biominer.detection.cropper import crop_with_padding
from biominer.detection.detector_base import DecodedImage
from biominer.detection.policy import DetectionPolicy, detection_is_bioclip_eligible
from biominer.detection.schema import DETECTION_OUTPUT_SCHEMA
from biominer.detection.segmentation import (
    NoneSegmenter,
    SegmentationUnavailable,
    Segmenter,
    detector_crop_mask_available,
    detector_masked_crop_bytes,
)
from biominer.species.context import SpeciesContext
from biominer.storage.parquet import write_parquet, write_parquet_batches
from biominer.vision.gates import BioClipGatePolicy, bioclip_score_input_decision
from biominer.vision.score_inputs import score_item_gate_fields, visual_input_id_for


PRIMARY_VISUAL_CLASSIFIER = "bioclip_object"
OBJECT_VISUAL_MODES: tuple[str, ...] = ("whole_image", "detector_crop", "detector_crop_segmentation")
AblationMode = Literal["whole_image", "detector_crop", "detector_crop_segmentation"]
TARGET_SCOPE_CANDIDATE_SELECTION_MODE = "taxon_scope_or_species_context"
TARGET_SCOPE_SPECIES_RERANK_STRATEGY = "first_pass_top20"
OBJECT_SCORE_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "detection_id": pl.String,
    "crop_hash": pl.String,
    "visual_input_id": pl.String,
    "visual_input_kind": pl.String,
    "bioclip_gate_mode": pl.String,
    "bioclip_gate_reason": pl.String,
    "model_id": pl.String,
    "model_version": pl.String,
    "model_checkpoint": pl.String,
    "candidate_set_id": pl.String,
    "classified_at": pl.String,
    "classification_mode": pl.String,
    "candidate_selection_mode": pl.String,
    "candidate_source": pl.String,
    "taxonomy_table_version": FIVE_RANK_OBJECT_SCORE_SCHEMA_EXTENSIONS["taxonomy_table_version"],
    "taxonomy_prompt_variant_version": FIVE_RANK_OBJECT_SCORE_SCHEMA_EXTENSIONS["taxonomy_prompt_variant_version"],
    "ablation_mode": pl.String,
    "species_first_pass_top_k": pl.Int64,
    "species_rerank_top_k": pl.Int64,
    "species_rerank_strategy": pl.String,
    "triage_group_top": pl.String,
    "triage_group_scores": pl.Struct({"butterfly_like": pl.Float64}),
    "family_top3": pl.List(pl.String),
    "family_top3_accepted_taxon_keys": FIVE_RANK_OBJECT_SCORE_SCHEMA_EXTENSIONS["family_top3_accepted_taxon_keys"],
    "family_top3_scores": FIVE_RANK_OBJECT_SCORE_SCHEMA_EXTENSIONS["family_top3_scores"],
    "family_top1": pl.String,
    "family_top1_score": pl.Float64,
    "family_margin": pl.Float64,
    "selected_family_key": FIVE_RANK_OBJECT_SCORE_SCHEMA_EXTENSIONS["selected_family_key"],
    "selected_family": FIVE_RANK_OBJECT_SCORE_SCHEMA_EXTENSIONS["selected_family"],
    "genus_top8": pl.List(pl.String),
    "genus_top1": pl.String,
    "genus_top1_score": pl.Float64,
    "genus_margin": pl.Float64,
    "species_candidate_family_key": FIVE_RANK_OBJECT_SCORE_SCHEMA_EXTENSIONS["species_candidate_family_key"],
    "species_candidate_family": FIVE_RANK_OBJECT_SCORE_SCHEMA_EXTENSIONS["species_candidate_family"],
    "species_candidate_count": FIVE_RANK_OBJECT_SCORE_SCHEMA_EXTENSIONS["species_candidate_count"],
    "species_top20": pl.List(pl.String),
    "species_top20_accepted_taxon_keys": pl.List(pl.String),
    "species_top20_scores": FIVE_RANK_OBJECT_SCORE_SCHEMA_EXTENSIONS["species_top20_scores"],
    "species_top5": pl.List(pl.String),
    "species_top5_accepted_taxon_keys": pl.List(pl.String),
    "species_top5_scores": FIVE_RANK_OBJECT_SCORE_SCHEMA_EXTENSIONS["species_top5_scores"],
    "species_top1": pl.String,
    "species_top1_scientific_name": pl.String,
    "species_top1_accepted_taxon_key": pl.String,
    "accepted_taxon_key": pl.String,
    "species_top1_score": pl.Float64,
    "species_top1_margin": pl.Float64,
    "target_accepted_taxon_key": pl.String,
    "target_species_score": pl.Float64,
    "target_species_rank": pl.Int64,
    "geospatial_prior_score": pl.Float64,
    "geospatial_prior_reason": pl.String,
    "text_evidence_score": pl.Float64,
    "comment_evidence_score": pl.Float64,
    "is_target_positive": pl.Boolean,
    "is_negative_material": pl.Boolean,
    "occurrence_bin": pl.String,
    "bin_reason": pl.String,
}
OBJECT_SCORE_OUTPUT_SCHEMA.update(FIVE_RANK_OBJECT_SCORE_SCHEMA_EXTENSIONS)
for _cascade_field, _cascade_dtype in PATH_CASCADE_OUTPUT_SCHEMA.items():
    OBJECT_SCORE_OUTPUT_SCHEMA.setdefault(_cascade_field, _cascade_dtype)
OBJECT_EVIDENCE_JOINED_SCHEMA: dict[str, pl.DataType] = {
    **OBJECT_SCORE_OUTPUT_SCHEMA,
    **DETECTION_OUTPUT_SCHEMA,
    "comments_fetched": pl.Boolean,
    "comment_count": pl.Int64,
    "species_match_from_comments": pl.Boolean,
    "species_name_from_comments": pl.String,
    "common_name_from_comments": pl.String,
    "life_stage_from_comments": pl.String,
    "date_evidence_from_comments": pl.String,
    "geo_evidence_from_comments": pl.String,
    "location_text_from_comments": pl.String,
    "comment_review_decision": pl.String,
    "comment_review_reason": pl.String,
    "flickr_text_species_candidate": pl.String,
    "bioclip_species_candidate": pl.String,
    "bioclip_tag_conflict": pl.Boolean,
    "comment_species_candidate": pl.String,
    "comment_resolves_conflict": pl.Boolean,
}
PHOTO_EVIDENCE_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "best_detection_id": pl.String,
    "detection_count": pl.Int64,
    "best_object_occurrence_bin": pl.String,
    "best_object_species_top1": pl.String,
    "best_object_score": pl.Float64,
    "photo_occurrence_bin": pl.String,
    "photo_bin_reason": pl.String,
    "all_detection_ids": pl.List(pl.String),
    "all_candidate_species": pl.List(pl.String),
    "all_selected_families": pl.List(pl.String),
    "all_selected_genera": pl.List(pl.String),
    "photo_selected_family": pl.String,
    "photo_selected_family_node_id": pl.String,
    "photo_selected_genus": pl.String,
    "photo_selected_genus_node_id": pl.String,
    "photo_species_top1": pl.String,
    "photo_species_top1_key": pl.String,
    "photo_species_confidence_score": pl.Float64,
    "photo_species_margin": pl.Float64,
    "photo_multi_object_conflict": pl.Boolean,
    "photo_review_reason": pl.String,
}


class ObjectBioClipScorer(Protocol):
    model_id: str
    model_version: str
    model_checkpoint: str

    def score(self, item: dict[str, Any], labels: tuple[str, ...]) -> dict[str, float]:
        ...


@dataclass(frozen=True)
class GeospatialPrior:
    score: float
    reason: str
    route_to_review: bool = False
    hard_discard: bool = False


@dataclass(frozen=True)
class ObjectScreenResult:
    frame: pl.DataFrame
    output_path: Path | None
    records_seen: int
    detections_seen: int
    crops_scored: int
    score_batches_written: int = 0
    adaptive_batching_enabled: bool = False
    bioclip_batch_retries: int = 0
    bioclip_batch_size_initial: int = 24
    bioclip_batch_size_final: int = 24
    bioclip_batch_size_min: int = 1
    segmentation_unavailable_count: int = 0
    segmentation_unavailable_reason: str | None = None
    segmentation_status: str | None = None
    visual_classifier: str = PRIMARY_VISUAL_CLASSIFIER
    visual_mode: str | None = None
    visual_mode_status: str | None = None


@dataclass(frozen=True)
class ObjectEvidenceOutputs:
    object_evidence_joined: Path
    photo_evidence_summary: Path


@dataclass(frozen=True)
class _ObjectScoringLabels:
    family: tuple[str, ...]
    genus: tuple[str, ...]
    species: tuple[str, ...]


@dataclass(frozen=True)
class MaterializedCropInputs:
    rows: list[dict[str, Any]]
    crop_path_by_hash: dict[str, Path]
    temp_dir: Path

    def cleanup(self) -> None:
        if self.temp_dir.exists():
            rmtree(self.temp_dir)


@dataclass(frozen=True)
class MaterializedCropBatch:
    items: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    crop_path_by_hash: dict[str, Path]
    temp_dir: Path
    retain_debug_crops: bool = False

    @property
    def crop_paths(self) -> list[Path]:
        return list(self.crop_path_by_hash.values())

    def cleanup(self, *, force: bool = False) -> None:
        if self.temp_dir.exists() and (force or not self.retain_debug_crops):
            rmtree(self.temp_dir)


@dataclass(frozen=True)
class DetectorCropMaterializationConfig:
    image_loader: Any
    temp_dir: Path
    crop_padding_ratio: float
    crop_target_px: int
    retain_debug_crops: bool


class FakeObjectBioClipScorer:
    model_id = "fake-bioclip"
    model_version = "test"
    model_checkpoint = "fake-checkpoint"

    def __init__(self, scores_by_crop: dict[str, dict[str, float]]) -> None:
        self.scores_by_crop = scores_by_crop

    def score(self, item: dict[str, Any], labels: tuple[str, ...]) -> dict[str, float]:
        scores = self.scores_by_crop.get(str(item.get("crop_hash") or ""), {})
        return {label: float(scores.get(label, 0.0)) for label in labels}

    def score_label_sets_batch(
        self,
        items: Sequence[dict[str, Any]],
        label_sets: Mapping[str, Sequence[str]],
    ) -> dict[str, list[dict[str, float]]]:
        return {
            str(name): [self.score(item, tuple(str(label) for label in labels)) for item in items]
            for name, labels in label_sets.items()
        }


class EphemeralCropBioClipScorer:
    def __init__(
        self,
        *,
        scorer: Any,
        image_loader: Any,
        temp_dir: str | Path,
        crop_padding_ratio: float = 0.12,
        crop_target_px: int = 336,
        model_id: str,
        model_version: str,
        model_checkpoint: str,
        retain_debug_crops: bool = False,
        debug_crop_limit: int = 500,
        segmenter: Segmenter | None = None,
    ) -> None:
        self._scorer = scorer
        self._image_loader = image_loader
        self._temp_dir = Path(temp_dir)
        self._crop_padding_ratio = crop_padding_ratio
        self._crop_target_px = crop_target_px
        self.model_id = model_id
        self.model_version = model_version
        self.model_checkpoint = model_checkpoint
        self._retain_debug_crops = retain_debug_crops
        self._debug_crop_limit = debug_crop_limit
        self._debug_crops_written = 0
        self._segmenter = segmenter or NoneSegmenter()

    def score(self, item: dict[str, Any], labels: tuple[str, ...]) -> dict[str, float]:
        mode = _ablation_mode(item)
        materialized_path = _materialized_detector_crop_path(item=item, mode=mode)
        if materialized_path is not None:
            return {str(label): float(score) for label, score in dict(self._scorer(materialized_path, labels)).items()}
        image = self._image_loader(item)
        if not isinstance(image, DecodedImage):
            raise TypeError("image_loader must return a DecodedImage")
        data, width, height, content_hash = self._visual_input_for_mode(item=item, image=image, mode=mode)
        crop_path, retained = self._write_temp_ppm_for_score(data, width=width, height=height, crop_hash=f"{mode}:{content_hash}")
        try:
            return {str(label): float(score) for label, score in dict(self._scorer(crop_path, labels)).items()}
        finally:
            if not retained:
                crop_path.unlink(missing_ok=True)

    def score_label_sets_batch(
        self,
        items: Sequence[dict[str, Any]],
        label_sets: Mapping[str, Sequence[str]],
    ) -> dict[str, list[dict[str, float]]]:
        crop_paths: list[Path] = []
        retained_paths: set[Path] = set()
        owned_paths: set[Path] = set()
        try:
            for item in items:
                mode = _ablation_mode(item)
                materialized_path = _materialized_detector_crop_path(item=item, mode=mode)
                if materialized_path is not None:
                    crop_path = materialized_path
                    retained = True
                else:
                    image = self._image_loader(item)
                    if not isinstance(image, DecodedImage):
                        raise TypeError("image_loader must return a DecodedImage")
                    data, width, height, content_hash = self._visual_input_for_mode(item=item, image=image, mode=mode)
                    crop_path, retained = self._write_temp_ppm_for_score(
                        data,
                        width=width,
                        height=height,
                        crop_hash=f"{mode}:{content_hash}",
                    )
                    owned_paths.add(crop_path)
                crop_paths.append(crop_path)
                if retained:
                    retained_paths.add(crop_path)

            label_sets_by_name = {str(name): tuple(str(label) for label in labels) for name, labels in label_sets.items()}
            batch_scorer = getattr(self._scorer, "score_label_sets_batch", None)
            if callable(batch_scorer):
                return _coerce_label_set_batch_scores(batch_scorer(crop_paths, label_sets_by_name), label_sets_by_name, len(crop_paths))
            score_batch = getattr(self._scorer, "score_batch", None)
            if callable(score_batch):
                return {
                    name: _coerce_score_batch(score_batch(crop_paths, labels), expected_count=len(crop_paths))
                    for name, labels in label_sets_by_name.items()
                }
            return {
                name: [
                    {str(label): float(score) for label, score in dict(self._scorer(path, labels)).items()}
                    for path in crop_paths
                ]
                for name, labels in label_sets_by_name.items()
            }
        finally:
            for crop_path in crop_paths:
                if crop_path in owned_paths and crop_path not in retained_paths:
                    crop_path.unlink(missing_ok=True)

    def supports_detector_crop_segmentation(self, item: dict[str, Any]) -> bool:
        return detector_crop_mask_available(item) or not isinstance(self._segmenter, NoneSegmenter)

    def embed_image_items(self, items: Sequence[dict[str, Any]]) -> list[list[float]]:
        embedder = getattr(self._scorer, "embed_image_paths", None)
        if not callable(embedder):
            raise ValueError("underlying BioCLIP scorer does not support image embeddings")
        crop_paths: list[Path] = []
        owned_paths: set[Path] = set()
        try:
            for item in items:
                mode = _ablation_mode(item)
                materialized_path = _materialized_detector_crop_path(item=item, mode=mode)
                if materialized_path is not None:
                    crop_paths.append(materialized_path)
                    continue
                image = self._image_loader(item)
                if not isinstance(image, DecodedImage):
                    raise TypeError("image_loader must return a DecodedImage")
                data, width, height, content_hash = self._visual_input_for_mode(item=item, image=image, mode=mode)
                crop_path, retained = self._write_temp_ppm_for_score(
                    data,
                    width=width,
                    height=height,
                    crop_hash=f"{mode}:{content_hash}",
                )
                crop_paths.append(crop_path)
                if not retained:
                    owned_paths.add(crop_path)
            return [[float(value) for value in embedding] for embedding in embedder(crop_paths)]
        finally:
            for crop_path in owned_paths:
                crop_path.unlink(missing_ok=True)

    def embed_text_labels(self, labels: Sequence[str]) -> list[list[float]]:
        embedder = getattr(self._scorer, "embed_text_labels", None)
        if not callable(embedder):
            raise ValueError("underlying BioCLIP scorer does not support text embeddings")
        return [
            [float(value) for value in embedding]
            for embedding in embedder(tuple(str(label) for label in labels))
        ]

    def detector_crop_materialization_config(self) -> DetectorCropMaterializationConfig:
        return DetectorCropMaterializationConfig(
            image_loader=self._image_loader,
            temp_dir=self._temp_dir,
            crop_padding_ratio=self._crop_padding_ratio,
            crop_target_px=self._crop_target_px,
            retain_debug_crops=self._retain_debug_crops,
        )

    def _visual_input_for_mode(self, *, item: dict[str, Any], image: DecodedImage, mode: AblationMode) -> tuple[bytes, int, int, str]:
        if mode == "whole_image":
            return image.data, image.width, image.height, _bytes_hash(image.data)

        bbox = item.get("bbox_xyxy")
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            raise ValueError("object BioCLIP crop scoring requires bbox_xyxy")
        crop = crop_with_padding(
            image,
            bbox_xyxy=tuple(float(value) for value in bbox),  # type: ignore[arg-type]
            padding_ratio=self._crop_padding_ratio,
            target_px=self._crop_target_px,
        )
        if mode == "detector_crop":
            return crop.encoded_bytes, crop.crop_width, crop.crop_height, crop.crop_hash

        masked = detector_masked_crop_bytes(item, crop)
        if masked is not None:
            return masked, crop.crop_width, crop.crop_height, _bytes_hash(masked)

        segmented = self._segmenter.segment_crop(crop)
        if segmented is None:
            raise SegmentationUnavailable("detector_masks_missing")
        return segmented, crop.crop_width, crop.crop_height, _bytes_hash(segmented)

    def _write_temp_ppm(self, data: bytes, *, width: int, height: int, crop_hash: str) -> Path:
        path, _retained = self._write_temp_ppm_for_score(data, width=width, height=height, crop_hash=crop_hash)
        return path

    def _write_temp_ppm_for_score(self, data: bytes, *, width: int, height: int, crop_hash: str) -> tuple[Path, bool]:
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        safe_hash = crop_hash.replace(":", "_").replace("/", "_")
        if self._should_retain_debug_crop():
            path = self._temp_dir / f"{safe_hash}.ppm"
            path.write_bytes(_ppm_bytes(data, width=width, height=height))
            self._debug_crops_written += 1
            return path, True
        handle = tempfile.NamedTemporaryFile(prefix=f"{safe_hash}_", suffix=".ppm", dir=self._temp_dir, delete=False)
        try:
            handle.write(_ppm_bytes(data, width=width, height=height))
            return Path(handle.name), False
        finally:
            handle.close()

    def _should_retain_debug_crop(self) -> bool:
        return self._retain_debug_crops and self._debug_crops_written < self._debug_crop_limit


class CachedObjectEmbeddingScorer:
    def __init__(
        self,
        *,
        text_embeddings: pl.DataFrame,
        image_embeddings: pl.DataFrame,
        candidate_set_id: str,
        model_id: str,
        model_version: str,
        model_checkpoint: str,
    ) -> None:
        self.model_id = model_id
        self.model_version = model_version
        self.model_checkpoint = model_checkpoint
        self._text_by_label = _cached_text_embedding_by_label(
            text_embeddings,
            candidate_set_id=candidate_set_id,
            model_id=model_id,
            model_checkpoint=model_checkpoint,
        )
        self._image_by_crop_hash = _cached_image_embedding_by_crop_hash(
            image_embeddings,
            model_id=model_id,
            model_checkpoint=model_checkpoint,
        )

    def score(self, item: dict[str, Any], labels: tuple[str, ...]) -> dict[str, float]:
        mode = str(item.get("ablation_mode") or "detector_crop")
        if mode != "detector_crop":
            raise ValueError("cached object image embeddings are only valid for detector_crop scoring")
        crop_hash = str(item.get("crop_hash") or "")
        try:
            image_embedding = self._image_by_crop_hash[crop_hash]
        except KeyError as exc:
            raise KeyError(f"missing cached object image embedding for crop_hash={crop_hash!r}") from exc
        scores: dict[str, float] = {}
        for label in labels:
            try:
                text_embedding = self._text_by_label[str(label)]
            except KeyError as exc:
                raise KeyError(f"missing cached candidate text embedding for label={label!r}") from exc
            scores[str(label)] = _cosine_similarity(image_embedding, text_embedding)
        return scores


def materialize_detector_crop_inputs(
    *,
    canonical_records: pl.DataFrame,
    detections: pl.DataFrame,
    image_loader: Any,
    temp_dir: str | Path,
    detection_policy: DetectionPolicy | None = None,
    bioclip_gate_policy: BioClipGatePolicy | None = None,
    crop_padding_ratio: float = 0.12,
    crop_target_px: int = 336,
) -> MaterializedCropInputs:
    gate_policy = _active_bioclip_gate_policy(
        detection_policy=detection_policy,
        bioclip_gate_policy=bioclip_gate_policy,
        detected_visual_input_kind="detector_crop",
    )
    records_by_photo = {
        (str(row.get("source") or ""), str(row.get("flickr_photo_id") or "")): row
        for row in canonical_records.to_dicts()
    }
    base = Path(temp_dir) / ".object_image_embedding_cache.tmp"
    if base.exists():
        rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    crop_path_by_hash: dict[str, Path] = {}
    try:
        for detection in detections.to_dicts():
            if not _score_detector_crop_decision(detection, gate_policy=gate_policy):
                continue
            key = (str(detection.get("source") or ""), str(detection.get("flickr_photo_id") or ""))
            record = _canonical_record_for_detection(records_by_photo, key=key)
            item = {**detection, **record, "ablation_mode": "detector_crop"}
            image = image_loader(item)
            if not isinstance(image, DecodedImage):
                raise TypeError("image_loader must return a DecodedImage")
            bbox = detection.get("bbox_xyxy")
            if not isinstance(bbox, list | tuple) or len(bbox) != 4:
                raise ValueError("object image embedding cache requires bbox_xyxy")
            crop = crop_with_padding(
                image,
                bbox_xyxy=tuple(float(value) for value in bbox),  # type: ignore[arg-type]
                padding_ratio=crop_padding_ratio,
                target_px=crop_target_px,
            )
            crop_hash = str(detection.get("crop_hash") or crop.crop_hash)
            rows.append(
                {
                    "source": str(detection.get("source") or ""),
                    "flickr_photo_id": str(detection.get("flickr_photo_id") or ""),
                    "detection_id": str(detection.get("detection_id") or ""),
                    "crop_hash": crop_hash,
                }
            )
            if crop_hash not in crop_path_by_hash:
                path = base / f"{_safe_file_stem(crop_hash)}.ppm"
                path.write_bytes(_ppm_bytes(crop.encoded_bytes, width=crop.crop_width, height=crop.crop_height))
                crop_path_by_hash[crop_hash] = path
    except Exception:
        if base.exists():
            rmtree(base)
        raise
    return MaterializedCropInputs(rows=rows, crop_path_by_hash=crop_path_by_hash, temp_dir=base)


def iter_materialized_detector_crop_batches(
    *,
    canonical_records: pl.DataFrame,
    detections: pl.DataFrame,
    image_loader: Any,
    temp_dir: str | Path,
    detection_policy: DetectionPolicy | None = None,
    bioclip_gate_policy: BioClipGatePolicy | None = None,
    crop_batch_size: int = 24,
    crop_padding_ratio: float = 0.08,
    crop_target_px: int = 336,
    retain_debug_crops: bool = False,
    cleanup_after_yield: bool = True,
) -> Iterator[MaterializedCropBatch]:
    if crop_batch_size <= 0:
        raise ValueError("crop_batch_size must be positive")
    gate_policy = _active_bioclip_gate_policy(
        detection_policy=detection_policy,
        bioclip_gate_policy=bioclip_gate_policy,
        detected_visual_input_kind="detector_crop",
    )
    records_by_photo = {
        (str(row.get("source") or ""), str(row.get("flickr_photo_id") or "")): row
        for row in canonical_records.to_dicts()
    }
    pending: list[dict[str, Any]] = []
    batch_index = 0
    for detection in detections.to_dicts():
        if not _score_detector_crop_decision(detection, gate_policy=gate_policy):
            continue
        pending.append(detection)
        if len(pending) >= crop_batch_size:
            yield from _yield_materialized_detector_crop_batch(
                detections=pending,
                records_by_photo=records_by_photo,
                image_loader=image_loader,
                temp_dir=temp_dir,
                batch_index=batch_index,
                gate_policy=gate_policy,
                crop_padding_ratio=crop_padding_ratio,
                crop_target_px=crop_target_px,
                retain_debug_crops=retain_debug_crops,
                cleanup_after_yield=cleanup_after_yield,
            )
            pending = []
            batch_index += 1
    if pending:
        yield from _yield_materialized_detector_crop_batch(
            detections=pending,
            records_by_photo=records_by_photo,
            image_loader=image_loader,
            temp_dir=temp_dir,
            batch_index=batch_index,
            gate_policy=gate_policy,
            crop_padding_ratio=crop_padding_ratio,
            crop_target_px=crop_target_px,
            retain_debug_crops=retain_debug_crops,
            cleanup_after_yield=cleanup_after_yield,
        )


def _yield_materialized_detector_crop_batch(
    *,
    detections: list[dict[str, Any]],
    records_by_photo: dict[tuple[str, str], dict[str, Any]],
    image_loader: Any,
    temp_dir: str | Path,
    batch_index: int,
    gate_policy: BioClipGatePolicy,
    crop_padding_ratio: float,
    crop_target_px: int,
    retain_debug_crops: bool,
    cleanup_after_yield: bool,
) -> Iterator[MaterializedCropBatch]:
    materialized = _materialize_detector_crop_batch(
        detections=detections,
        records_by_photo=records_by_photo,
        image_loader=image_loader,
        temp_dir=temp_dir,
        batch_index=batch_index,
        gate_policy=gate_policy,
        crop_padding_ratio=crop_padding_ratio,
        crop_target_px=crop_target_px,
        retain_debug_crops=retain_debug_crops,
    )
    try:
        yield materialized
    finally:
        if cleanup_after_yield:
            materialized.cleanup()


def _materialize_detector_crop_batch(
    *,
    detections: list[dict[str, Any]],
    records_by_photo: dict[tuple[str, str], dict[str, Any]],
    image_loader: Any,
    temp_dir: str | Path,
    batch_index: int,
    gate_policy: BioClipGatePolicy,
    crop_padding_ratio: float,
    crop_target_px: int,
    retain_debug_crops: bool,
) -> MaterializedCropBatch:
    root = Path(temp_dir)
    root.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix=f".object_bioclip_crops_{batch_index:06d}_", dir=root))
    rows: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    crop_path_by_hash: dict[str, Path] = {}
    image_by_photo: dict[tuple[str, str], DecodedImage] = {}
    try:
        for detection in detections:
            decision = bioclip_score_input_decision(detection, gate_policy)
            if not decision.should_score or decision.visual_input_kind != "detector_crop":
                continue
            key = (str(detection.get("source") or ""), str(detection.get("flickr_photo_id") or ""))
            record = _canonical_record_for_detection(records_by_photo, key=key)
            item = {**detection, **record, "ablation_mode": "detector_crop"}
            image = image_by_photo.get(key)
            if image is None:
                loaded = image_loader(item)
                if not isinstance(loaded, DecodedImage):
                    raise TypeError("image_loader must return a DecodedImage")
                image_by_photo[key] = loaded
                image = loaded
            bbox = detection.get("bbox_xyxy")
            if not isinstance(bbox, list | tuple) or len(bbox) != 4:
                raise ValueError("object BioCLIP crop batch requires bbox_xyxy")
            crop = crop_with_padding(
                image,
                bbox_xyxy=tuple(float(value) for value in bbox),  # type: ignore[arg-type]
                padding_ratio=crop_padding_ratio,
                target_px=crop_target_px,
            )
            crop_hash = str(detection.get("crop_hash") or crop.crop_hash)
            visual_input_id = visual_input_id_for(
                source=str(detection.get("source") or ""),
                flickr_photo_id=str(detection.get("flickr_photo_id") or ""),
                detection_id=str(detection.get("detection_id") or ""),
                visual_input_kind="detector_crop",
                crop_hash=crop_hash,
            )
            path = crop_path_by_hash.get(crop_hash)
            if path is None:
                path = base / f"{len(crop_path_by_hash):06d}_{_safe_file_stem(crop_hash)}.ppm"
                path.write_bytes(_ppm_bytes(crop.encoded_bytes, width=crop.crop_width, height=crop.crop_height))
                crop_path_by_hash[crop_hash] = path
            row = {
                "source": str(detection.get("source") or ""),
                "flickr_photo_id": str(detection.get("flickr_photo_id") or ""),
                "detection_id": str(detection.get("detection_id") or ""),
                "crop_hash": crop_hash,
                "crop_path": str(path),
                "materialized_crop_hash": crop.crop_hash,
                "visual_input_id": visual_input_id,
                "visual_input_kind": "detector_crop",
                "bioclip_gate_mode": decision.bioclip_gate_mode,
                "bioclip_gate_reason": decision.bioclip_gate_reason,
            }
            rows.append(row)
            items.append(
                {
                    **item,
                    "ablation_mode": "detector_crop",
                    **score_item_gate_fields(decision=decision, visual_input_id=visual_input_id, crop_hash=crop_hash),
                    "crop_hash": crop_hash,
                    "crop_path": path,
                    "materialized_crop_hash": crop.crop_hash,
                    "crop_padding_ratio": crop_padding_ratio,
                    "crop_width": crop.crop_width,
                    "crop_height": crop.crop_height,
                    "clamped_bbox_xyxy": crop.clamped_bbox_xyxy,
                    "padded_bbox_xyxy": crop.padded_bbox_xyxy,
                    "crop_storage_policy": crop.storage_policy,
                }
            )
    except Exception:
        if base.exists():
            rmtree(base)
        raise
    return MaterializedCropBatch(
        items=items,
        rows=rows,
        crop_path_by_hash=crop_path_by_hash,
        temp_dir=base,
        retain_debug_crops=retain_debug_crops,
    )


def screen_object_detections(
    *,
    canonical_records: pl.DataFrame,
    detections: pl.DataFrame,
    species_context: SpeciesContext,
    candidate_set: CandidateSet,
    scorer: ObjectBioClipScorer,
    output_path: str | Path | None = None,
    ablation_mode: AblationMode = "detector_crop",
    detection_policy: DetectionPolicy | None = None,
    bioclip_gate_policy: BioClipGatePolicy | None = None,
    geo_prior_table: pl.DataFrame | None = None,
    parquet_batch_rows: int = 10000,
    bioclip_batch_size: int = 24,
    adaptive_batching: bool = False,
    min_bioclip_batch_size: int = 1,
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    family_top_k: int = DEFAULT_FAMILY_TOP_K,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
    path_taxonomy_store: PathTaxonomyStore | None = None,
    taxonomy_text_embedding_index: TaxonomyTextEmbeddingIndex | None = None,
) -> ObjectScreenResult:
    if bioclip_batch_size <= 0:
        raise ValueError("bioclip_batch_size must be positive")
    if min_bioclip_batch_size <= 0:
        raise ValueError("min_bioclip_batch_size must be positive")
    classification_mode = normalize_classification_mode(classification_mode)
    if (path_taxonomy_store is None) != (taxonomy_text_embedding_index is None):
        raise ValueError(
            "path_taxonomy_store and taxonomy_text_embedding_index are required together "
            "for classification-v3 hierarchical scoring"
        )
    if classification_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION and path_taxonomy_store is None:
        raise ValueError(
            "classification-v3 path_taxonomy_store and taxonomy_text_embedding_index "
            "are required for hierarchical_butterfly_classification"
        )
    family_top_k, species_first_pass_top_k, species_rerank_top_k = _validate_visual_top_k(
        family_top_k=family_top_k,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
    )
    if classification_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION:
        validate_production_cascade_settings(
            beam_strategy=GLOBAL_RANK_TOP_K_BEAM_STRATEGY,
            rank_beam_width=family_top_k,
            species_first_pass_top_k=species_first_pass_top_k,
            species_rerank_top_k=species_rerank_top_k,
            species_report_top_k=DEFAULT_SPECIES_REPORT_TOP_K,
        )
    gate_policy = _active_bioclip_gate_policy(
        detection_policy=detection_policy,
        bioclip_gate_policy=bioclip_gate_policy,
        detected_visual_input_kind=ablation_mode,
    )
    records_by_photo = {
        (str(row.get("source") or ""), str(row.get("flickr_photo_id") or "")): row
        for row in canonical_records.to_dicts()
    }
    rows: list[dict[str, Any]] = []
    output = Path(output_path) if output_path is not None else None
    batch_dir = _prepare_score_batch_dir(output) if output is not None else None
    row_buffer: list[dict[str, Any]] = []
    batch_paths: list[Path] = []
    crops_scored = 0
    segmentation_unavailable_count = 0
    segmentation_unavailable_reason: str | None = None
    score_items: list[dict[str, Any]] = []
    active_bioclip_batch_size = 1 if ablation_mode == "detector_crop_segmentation" else max(1, bioclip_batch_size)
    if min_bioclip_batch_size > active_bioclip_batch_size:
        raise ValueError("min_bioclip_batch_size must be <= active BioCLIP batch size")
    current_bioclip_batch_size = active_bioclip_batch_size
    bioclip_batch_retries = 0
    materialized_batches_to_cleanup: list[MaterializedCropBatch] = []
    committed_output = False
    materialized_crop_config = (
        _detector_crop_materialization_config(scorer)
        if ablation_mode == "detector_crop"
        else None
    )

    def score_batch(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if classification_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION:
            assert path_taxonomy_store is not None
            assert taxonomy_text_embedding_index is not None
            return _score_hierarchical_detection_batch(
                items=items,
                scorer=scorer,
                path_taxonomy_store=path_taxonomy_store,
                taxonomy_text_embedding_index=taxonomy_text_embedding_index,
            )
        return _score_detection_batch(
            items=items,
            context=species_context,
            candidate_set=candidate_set,
            scorer=scorer,
            ablation_mode=ablation_mode,
            geo_prior_table=geo_prior_table,
            classification_mode=classification_mode,
            family_top_k=family_top_k,
            species_first_pass_top_k=species_first_pass_top_k,
            species_rerank_top_k=species_rerank_top_k,
        )

    def emit_score_rows(score_rows: list[dict[str, Any]]) -> None:
        nonlocal crops_scored
        crops_scored += len(score_rows)
        if output is None or batch_dir is None:
            rows.extend(score_rows)
            return
        for score_row in score_rows:
            _buffer_score_rows(
                [score_row],
                row_buffer=row_buffer,
                batch_paths=batch_paths,
                batch_dir=batch_dir,
                parquet_batch_rows=parquet_batch_rows,
            )

    def flush_score_items() -> None:
        nonlocal bioclip_batch_retries, current_bioclip_batch_size, segmentation_unavailable_count, segmentation_unavailable_reason
        if not score_items:
            return
        items = list(score_items)
        score_items.clear()
        pending = [items]
        while pending:
            batch = pending.pop(0)
            try:
                score_rows = score_batch(batch)
            except SegmentationUnavailable as exc:
                if ablation_mode != "detector_crop_segmentation":
                    raise
                segmentation_unavailable_count += len(batch)
                segmentation_unavailable_reason = segmentation_unavailable_reason or str(exc) or "detector_masks_missing"
                continue
            except RuntimeError as exc:
                if not _should_retry_bioclip_batch(
                    exc,
                    adaptive_batching=adaptive_batching,
                    batch_size=len(batch),
                    current_batch_size=current_bioclip_batch_size,
                    min_batch_size=min_bioclip_batch_size,
                ):
                    raise
                current_bioclip_batch_size = _next_bioclip_batch_size(
                    current_batch_size=current_bioclip_batch_size,
                    failed_batch_size=len(batch),
                    min_batch_size=min_bioclip_batch_size,
                )
                bioclip_batch_retries += 1
                pending = _chunks(batch, current_bioclip_batch_size) + pending
                continue
            emit_score_rows(score_rows)

    try:
        if materialized_crop_config is not None:
            for crop_batch in iter_materialized_detector_crop_batches(
                canonical_records=canonical_records,
                detections=detections,
                image_loader=materialized_crop_config.image_loader,
                temp_dir=materialized_crop_config.temp_dir,
                detection_policy=detection_policy,
                bioclip_gate_policy=gate_policy,
                crop_batch_size=active_bioclip_batch_size,
                crop_padding_ratio=materialized_crop_config.crop_padding_ratio,
                crop_target_px=materialized_crop_config.crop_target_px,
                retain_debug_crops=materialized_crop_config.retain_debug_crops,
                cleanup_after_yield=False,
            ):
                materialized_batches_to_cleanup.append(crop_batch)
                score_items.extend(crop_batch.items)
                flush_score_items()
            for item in _score_items_for_visual_input_kind(
                canonical_records_by_photo=records_by_photo,
                detections=detections,
                gate_policy=gate_policy,
                visual_input_kind="whole_image",
            ):
                score_items.append(item)
                if len(score_items) >= current_bioclip_batch_size:
                    flush_score_items()
        else:
            for detection in detections.to_dicts():
                decision = bioclip_score_input_decision(detection, gate_policy)
                if not decision.should_score:
                    continue
                key = (str(detection.get("source") or ""), str(detection.get("flickr_photo_id") or ""))
                record = _canonical_record_for_detection(records_by_photo, key=key)
                item_ablation_mode = decision.visual_input_kind or ablation_mode
                crop_hash = str(detection.get("crop_hash") or "")
                visual_input_id = visual_input_id_for(
                    source=str(detection.get("source") or ""),
                    flickr_photo_id=str(detection.get("flickr_photo_id") or ""),
                    detection_id=str(detection.get("detection_id") or ""),
                    visual_input_kind=str(item_ablation_mode),
                    crop_hash=crop_hash,
                )
                item = {
                    **detection,
                    **record,
                    **score_item_gate_fields(decision=decision, visual_input_id=visual_input_id, crop_hash=crop_hash),
                    "ablation_mode": item_ablation_mode,
                }
                if ablation_mode == "detector_crop_segmentation" and not _scorer_supports_detector_crop_segmentation(scorer, item):
                    segmentation_unavailable_count += 1
                    segmentation_unavailable_reason = segmentation_unavailable_reason or "detector_masks_missing"
                    continue
                score_items.append(item)
                if len(score_items) >= current_bioclip_batch_size:
                    flush_score_items()
        flush_score_items()
        if output is not None and batch_dir is not None:
            _flush_score_row_buffer(row_buffer=row_buffer, batch_paths=batch_paths, batch_dir=batch_dir)
            write_parquet_batches(
                (pl.read_parquet(path) for path in batch_paths),
                output,
                schema=OBJECT_SCORE_OUTPUT_SCHEMA,
            )
            frame = _ensure_columns(pl.read_parquet(output), OBJECT_SCORE_OUTPUT_SCHEMA)
        else:
            frame = _ensure_columns(pl.DataFrame(rows), OBJECT_SCORE_OUTPUT_SCHEMA) if rows else empty_object_score_frame()
        result = ObjectScreenResult(
            frame=frame,
            output_path=output,
            records_seen=canonical_records.height,
            detections_seen=detections.height,
            crops_scored=crops_scored,
            score_batches_written=len(batch_paths),
            adaptive_batching_enabled=bool(adaptive_batching),
            bioclip_batch_retries=bioclip_batch_retries,
            bioclip_batch_size_initial=active_bioclip_batch_size,
            bioclip_batch_size_final=current_bioclip_batch_size,
            bioclip_batch_size_min=min_bioclip_batch_size,
            segmentation_unavailable_count=segmentation_unavailable_count,
            segmentation_unavailable_reason=segmentation_unavailable_reason,
            segmentation_status=_segmentation_status(
                mode=ablation_mode,
                crops_scored=crops_scored,
                unavailable_count=segmentation_unavailable_count,
            ),
            visual_mode=ablation_mode,
            visual_mode_status=_visual_mode_status(
                mode=ablation_mode,
                crops_scored=crops_scored,
                unavailable_count=segmentation_unavailable_count,
            ),
        )
        committed_output = True
        return result
    finally:
        if committed_output:
            for crop_batch in materialized_batches_to_cleanup:
                crop_batch.cleanup()
        if batch_dir is not None and batch_dir.exists():
            rmtree(batch_dir)


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]


def _should_retry_bioclip_batch(
    exc: RuntimeError,
    *,
    adaptive_batching: bool,
    batch_size: int,
    current_batch_size: int,
    min_batch_size: int,
) -> bool:
    return (
        adaptive_batching
        and batch_size > min_batch_size
        and current_batch_size > min_batch_size
        and is_bioclip_memory_error(exc)
    )


def _next_bioclip_batch_size(
    *,
    current_batch_size: int,
    failed_batch_size: int,
    min_batch_size: int,
) -> int:
    if current_batch_size <= min_batch_size or failed_batch_size <= min_batch_size:
        return min_batch_size
    return max(min_batch_size, min(current_batch_size // 2, failed_batch_size // 2))


def is_bioclip_memory_error(exc: BaseException) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    message = " ".join(str(exc).casefold().split())
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cuda out of memory",
            "mps memory",
            "allocation failed",
        )
    )


def _prepare_score_batch_dir(output_path: Path) -> Path:
    batch_dir = output_path.parent / f".{output_path.name}.batches.tmp"
    if batch_dir.exists():
        rmtree(batch_dir)
    return batch_dir


def _buffer_score_rows(
    rows: list[dict[str, Any]],
    *,
    row_buffer: list[dict[str, Any]],
    batch_paths: list[Path],
    batch_dir: Path,
    parquet_batch_rows: int,
) -> None:
    row_buffer.extend(rows)
    if len(row_buffer) >= max(1, parquet_batch_rows):
        _flush_score_row_buffer(row_buffer=row_buffer, batch_paths=batch_paths, batch_dir=batch_dir)


def _flush_score_row_buffer(*, row_buffer: list[dict[str, Any]], batch_paths: list[Path], batch_dir: Path) -> None:
    if not row_buffer:
        return
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_path = batch_dir / f"batch-{len(batch_paths):06d}.parquet"
    write_parquet(pl.DataFrame(row_buffer), batch_path)
    batch_paths.append(batch_path)
    row_buffer.clear()


def empty_object_score_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=OBJECT_SCORE_OUTPUT_SCHEMA)


def object_score_audit_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "classification_mode_counts": _string_value_counts(frame, "classification_mode"),
        "candidate_selection_mode_counts": _string_value_counts(frame, "candidate_selection_mode"),
        "species_rerank_strategy_counts": _string_value_counts(frame, "species_rerank_strategy"),
        "taxonomy_table_versions": _string_values(frame, "taxonomy_table_version"),
        "taxonomy_prompt_variant_versions": _string_values(frame, "taxonomy_prompt_variant_version"),
        "selected_family_counts": _string_value_counts(frame, "selected_family"),
        "selected_family_key_counts": _string_value_counts(frame, "selected_family_key"),
        "species_top1_counts": _string_value_counts(frame, "species_top1_scientific_name"),
        "accepted_taxon_key_counts": _string_value_counts(frame, "accepted_taxon_key"),
        **_numeric_summary(frame, "species_candidate_count"),
    }


def _canonical_record_for_detection(
    records_by_photo: dict[tuple[str, str], dict[str, Any]],
    *,
    key: tuple[str, str],
) -> dict[str, Any]:
    record = records_by_photo.get(key)
    if record is None:
        source, photo_id = key
        raise ValueError(f"object BioCLIP detection has no canonical source record: source={source!r}, flickr_photo_id={photo_id!r}")
    return record


def apply_geospatial_soft_prior(
    row: dict[str, Any],
    candidate_or_context: CandidateTaxon | SpeciesContext,
    species_context: SpeciesContext | None = None,
    *,
    visual_score: float,
    geo_prior_table: pl.DataFrame | None = None,
) -> GeospatialPrior:
    candidate = candidate_or_context if isinstance(candidate_or_context, CandidateTaxon) else None
    context = species_context or candidate_or_context
    if not isinstance(context, SpeciesContext):
        raise TypeError("species_context is required when applying a candidate-specific geospatial prior")
    latitude = _optional_float(row.get("latitude"))
    longitude = _optional_float(row.get("longitude"))
    if latitude is None or longitude is None:
        return GeospatialPrior(score=0.0, reason="missing_geo")
    table_prior = _geo_prior_table_match(
        latitude=latitude,
        longitude=longitude,
        species_context=context,
        candidate=candidate,
        geo_prior_table=geo_prior_table,
    )
    if table_prior is not None:
        return table_prior
    for region in context.regions:
        if region.bbox and _coordinate_in_bbox(latitude=latitude, longitude=longitude, bbox=region.bbox):
            return GeospatialPrior(score=0.10, reason="within_context_region")
    if visual_score >= DEFAULT_BUCKET_POLICY.gold_species_threshold:
        return GeospatialPrior(score=-0.10, reason="geospatial_conflict", route_to_review=True)
    return GeospatialPrior(score=-0.05, reason="outside_context_region")


def write_object_evidence_outputs(
    *,
    canonical_records_path: str | Path,
    detections_path: str | Path,
    scores_path: str | Path,
    joined_output_path: str | Path,
    photo_summary_output_path: str | Path,
    species_context: SpeciesContext | None = None,
) -> ObjectEvidenceOutputs:
    canonical = pl.read_parquet(canonical_records_path)
    detections = pl.read_parquet(detections_path)
    scores = pl.read_parquet(scores_path)
    joined = _object_evidence_joined(canonical=canonical, detections=detections, scores=scores)
    summary = _photo_summary(scores, canonical=canonical, detections=detections, species_context=species_context)
    joined_path = write_parquet(joined, joined_output_path)
    summary_path = write_parquet(summary, photo_summary_output_path)
    return ObjectEvidenceOutputs(object_evidence_joined=joined_path, photo_evidence_summary=summary_path)


def _object_evidence_joined(*, canonical: pl.DataFrame, detections: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    join_keys = ["source", "flickr_photo_id", "detection_id"]
    scored = (
        scores.join(canonical, on=["source", "flickr_photo_id"], how="left", suffix="_canonical")
        .join(detections, on=join_keys, how="left", suffix="_detection")
        if not scores.is_empty() and _has_columns(scores, join_keys)
        else pl.DataFrame()
    )
    detection_only = detections
    if not scores.is_empty() and _has_columns(scores, ["source", "flickr_photo_id", "detection_id"]):
        scored_detection_keys = scores.select(["source", "flickr_photo_id", "detection_id"]).unique()
        detection_only = detections.join(scored_detection_keys, on=["source", "flickr_photo_id", "detection_id"], how="anti")
    if not detection_only.is_empty():
        detection_only = detection_only.join(canonical, on=["source", "flickr_photo_id"], how="left", suffix="_canonical")
    if scored.is_empty():
        return _ensure_columns(detection_only, OBJECT_EVIDENCE_JOINED_SCHEMA)
    if detection_only.is_empty():
        return _ensure_columns(scored, OBJECT_EVIDENCE_JOINED_SCHEMA)
    return _ensure_columns(pl.concat([scored, detection_only], how="diagonal_relaxed"), OBJECT_EVIDENCE_JOINED_SCHEMA)


def _score_detection(
    *,
    item: dict[str, Any],
    context: SpeciesContext,
    candidate_set: CandidateSet,
    scorer: ObjectBioClipScorer,
    ablation_mode: AblationMode,
    geo_prior_table: pl.DataFrame | None = None,
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    family_top_k: int = DEFAULT_FAMILY_TOP_K,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
) -> dict[str, Any]:
    classification_mode = normalize_classification_mode(classification_mode)
    _raise_if_hierarchical_classification(classification_mode)
    item_ablation_mode = _ablation_mode({**item, "ablation_mode": item.get("ablation_mode") or ablation_mode})
    family_top_k, species_first_pass_top_k, species_rerank_top_k = _validate_visual_top_k(
        family_top_k=family_top_k,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
    )
    labels = _object_scoring_labels(candidate_set)
    family_scores = scorer.score(item, labels.family) if labels.family else {}
    genus_scores = scorer.score(item, labels.genus) if labels.genus else {}
    species_scores = scorer.score(item, labels.species)
    ranked_species_top20 = _rank_species(candidate_set.species_candidates, species_scores)[:species_first_pass_top_k]
    ranked_families = _rank_labels(labels.family, family_scores) if labels.family else []
    family_top1 = ranked_families[0][0] if ranked_families else None
    family_filtered_ranked_species_top20 = _rank_species_for_family(
        candidates=candidate_set.species_candidates,
        ranked_species=ranked_species_top20,
        family=family_top1,
    )
    rerank_candidates = _species_rerank_candidates(
        candidate_set.species_candidates,
        ranked_species_top20=family_filtered_ranked_species_top20,
    )
    rerank_scores = scorer.score(item, _species_prompt_labels(rerank_candidates)) if rerank_candidates else {}
    return _score_detection_from_scores(
        item=item,
        context=context,
        candidate_set=candidate_set,
        scorer=scorer,
        ablation_mode=item_ablation_mode,
        labels=labels,
        family_scores=family_scores,
        genus_scores=genus_scores,
        ranked_species_full_top20=ranked_species_top20,
        ranked_species_top20=family_filtered_ranked_species_top20,
        rerank_candidates=rerank_candidates,
        rerank_scores=rerank_scores,
        geo_prior_table=geo_prior_table,
        classification_mode=classification_mode,
        family_top_k=family_top_k,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
    )


def _score_detection_batch(
    *,
    items: list[dict[str, Any]],
    context: SpeciesContext,
    candidate_set: CandidateSet,
    scorer: ObjectBioClipScorer,
    ablation_mode: AblationMode,
    geo_prior_table: pl.DataFrame | None = None,
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    family_top_k: int = DEFAULT_FAMILY_TOP_K,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
) -> list[dict[str, Any]]:
    if not items:
        return []
    classification_mode = normalize_classification_mode(classification_mode)
    _raise_if_hierarchical_classification(classification_mode)
    family_top_k, species_first_pass_top_k, species_rerank_top_k = _validate_visual_top_k(
        family_top_k=family_top_k,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
    )
    labels = _object_scoring_labels(candidate_set)
    initial_label_sets: dict[str, tuple[str, ...]] = {}
    if labels.family:
        initial_label_sets["family"] = labels.family
    if labels.genus:
        initial_label_sets["genus"] = labels.genus
    initial_label_sets["species"] = labels.species
    initial_scores = _score_label_sets_for_items(scorer, items, initial_label_sets)
    ranked_species_top20_by_index: list[list[tuple[str, float]]] = []
    family_ranked_species_top20_by_index: list[list[tuple[str, float]]] = []
    rerank_candidates_by_index: list[tuple[CandidateTaxon, ...]] = []
    rerank_scores_by_index: list[dict[str, float]] = [{} for _item in items]

    for index, _item in enumerate(items):
        ranked_species_top20 = _rank_species(candidate_set.species_candidates, initial_scores["species"][index])[
            :species_first_pass_top_k
        ]
        ranked_species_top20_by_index.append(ranked_species_top20)
        ranked_families = _rank_labels(labels.family, initial_scores["family"][index]) if labels.family else []
        family_top1 = ranked_families[0][0] if ranked_families else None
        family_ranked_species_top20 = _rank_species_for_family(
            candidates=candidate_set.species_candidates,
            ranked_species=ranked_species_top20,
            family=family_top1,
        )
        family_ranked_species_top20_by_index.append(family_ranked_species_top20)
        rerank_candidates_by_index.append(
            _species_rerank_candidates(
                candidate_set.species_candidates,
                ranked_species_top20=family_ranked_species_top20,
            )
        )

    rerank_groups: dict[tuple[str, ...], list[int]] = {}
    for index, rerank_candidates in enumerate(rerank_candidates_by_index):
        rerank_labels = _species_prompt_labels(rerank_candidates)
        if rerank_labels:
            rerank_groups.setdefault(rerank_labels, []).append(index)

    for rerank_labels, indices in rerank_groups.items():
        rerank_items = [items[index] for index in indices]
        rerank_scores = _score_label_sets_for_items(scorer, rerank_items, {"rerank": rerank_labels})["rerank"]
        for index, scores in zip(indices, rerank_scores, strict=True):
            rerank_scores_by_index[index] = scores

    rows: list[dict[str, Any]] = []
    empty_scores = [{} for _item in items]
    family_scores = initial_scores.get("family", empty_scores)
    genus_scores = initial_scores.get("genus", empty_scores)
    for index, item in enumerate(items):
        rows.append(
            _score_detection_from_scores(
                item=item,
                context=context,
                candidate_set=candidate_set,
                scorer=scorer,
                ablation_mode=_ablation_mode({**item, "ablation_mode": item.get("ablation_mode") or ablation_mode}),
                labels=labels,
                family_scores=family_scores[index],
                genus_scores=genus_scores[index],
                ranked_species_full_top20=ranked_species_top20_by_index[index],
                ranked_species_top20=family_ranked_species_top20_by_index[index],
                rerank_candidates=rerank_candidates_by_index[index],
                rerank_scores=rerank_scores_by_index[index],
                geo_prior_table=geo_prior_table,
                classification_mode=classification_mode,
                family_top_k=family_top_k,
                species_first_pass_top_k=species_first_pass_top_k,
                species_rerank_top_k=species_rerank_top_k,
            )
        )
    return rows


def _score_hierarchical_detection_batch(
    *,
    items: list[dict[str, Any]],
    scorer: ObjectBioClipScorer,
    path_taxonomy_store: PathTaxonomyStore,
    taxonomy_text_embedding_index: TaxonomyTextEmbeddingIndex,
) -> list[dict[str, Any]]:
    results = classify_path_cascade_batch(
        items=items,
        embedding_scorer=scorer,
        taxonomy_store=path_taxonomy_store,
        taxonomy_text_embedding_index=taxonomy_text_embedding_index,
    )
    return [
        path_cascade_result_to_object_score_row(
            item=item,
            result=result,
            scorer=scorer,
        )
        for item, result in zip(items, results, strict=True)
    ]


def _score_detection_from_scores(
    *,
    item: dict[str, Any],
    context: SpeciesContext,
    candidate_set: CandidateSet,
    scorer: ObjectBioClipScorer,
    ablation_mode: AblationMode,
    labels: _ObjectScoringLabels,
    family_scores: dict[str, float],
    genus_scores: dict[str, float],
    ranked_species_full_top20: list[tuple[str, float]],
    ranked_species_top20: list[tuple[str, float]],
    rerank_candidates: tuple[CandidateTaxon, ...],
    rerank_scores: dict[str, float],
    geo_prior_table: pl.DataFrame | None,
    classification_mode: ClassificationMode,
    family_top_k: int,
    species_first_pass_top_k: int,
    species_rerank_top_k: int,
) -> dict[str, Any]:
    ranked_families = _rank_labels(labels.family, family_scores)
    ranked_genera = _rank_labels(labels.genus, genus_scores)
    ranked_species = _rank_species(rerank_candidates, rerank_scores) if rerank_candidates else ranked_species_top20
    target_score = _target_score(ranked_species_full_top20, context.scientific_name)
    top1_name = ranked_species[0][0] if ranked_species else None
    species_top20 = [name for name, _score in ranked_species_top20]
    species_top5 = [name for name, _score in ranked_species[:species_rerank_top_k]]
    taxon_key_by_name = _taxon_key_by_name(candidate_set.species_candidates)
    top1_taxon_key = _taxon_key_for_name(taxon_key_by_name, top1_name)
    top1_candidate = _candidate_for_name(candidate_set.species_candidates, top1_name)
    top1_score = ranked_species[0][1] if ranked_species else 0.0
    target_rank = _target_rank(ranked_species_full_top20, context.scientific_name)
    margin = _margin(ranked_species)
    family_margin = _margin(ranked_families)
    genus_margin = _margin(ranked_genera)
    geo_candidate: CandidateTaxon | SpeciesContext = top1_candidate or context
    geo = apply_geospatial_soft_prior(
        item,
        geo_candidate,
        context if top1_candidate is not None else None,
        visual_score=top1_score if top1_candidate is not None else target_score,
        geo_prior_table=geo_prior_table,
    )
    bucket, reason = _bucket(item=item, target_score=target_score, target_rank=target_rank, margin=margin, geo=geo)
    negative_reason = _hard_negative_photo_reason(item)
    return {
        "source": str(item.get("source") or ""),
        "flickr_photo_id": str(item.get("flickr_photo_id") or ""),
        "detection_id": str(item.get("detection_id") or ""),
        "crop_hash": str(item.get("crop_hash") or ""),
        "visual_input_id": _visual_input_id(item, ablation_mode=ablation_mode),
        "visual_input_kind": str(item.get("visual_input_kind") or ablation_mode),
        "bioclip_gate_mode": _string_or_none(item.get("bioclip_gate_mode")),
        "bioclip_gate_reason": _string_or_none(item.get("bioclip_gate_reason")),
        "model_id": scorer.model_id,
        "model_version": scorer.model_version,
        "model_checkpoint": scorer.model_checkpoint,
        "candidate_set_id": candidate_set.candidate_set_id,
        "classified_at": datetime.now(UTC).isoformat(),
        "classification_mode": classification_mode,
        "candidate_selection_mode": TARGET_SCOPE_CANDIDATE_SELECTION_MODE,
        "candidate_source": ",".join(candidate_set.source_evidence),
        "ablation_mode": ablation_mode,
        "species_first_pass_top_k": int(species_first_pass_top_k),
        "species_rerank_top_k": int(species_rerank_top_k),
        "species_rerank_strategy": _target_scope_species_rerank_strategy(species_first_pass_top_k),
        "triage_group_top": "butterfly_like",
        "triage_group_scores": {"butterfly_like": float(item.get("detector_score") or 0.0)},
        "family_top3": [name for name, _score in ranked_families[:family_top_k]],
        "family_top1": ranked_families[0][0] if ranked_families else None,
        "family_top1_score": ranked_families[0][1] if ranked_families else 0.0,
        "family_margin": family_margin,
        "genus_top8": [name for name, _score in ranked_genera[:8]],
        "genus_top1": ranked_genera[0][0] if ranked_genera else None,
        "genus_top1_score": ranked_genera[0][1] if ranked_genera else 0.0,
        "genus_margin": genus_margin,
        "species_top20": species_top20,
        "species_top20_accepted_taxon_keys": [_taxon_key_for_name(taxon_key_by_name, name) for name in species_top20],
        "species_top5": species_top5,
        "species_top5_accepted_taxon_keys": [_taxon_key_for_name(taxon_key_by_name, name) for name in species_top5],
        "species_top1": top1_name,
        "species_top1_scientific_name": top1_name,
        "species_top1_accepted_taxon_key": top1_taxon_key,
        "accepted_taxon_key": top1_taxon_key,
        "species_top1_score": top1_score,
        "species_top1_margin": margin,
        "target_accepted_taxon_key": context.accepted_taxon_key,
        "target_species_score": target_score,
        "target_species_rank": target_rank,
        "geospatial_prior_score": geo.score,
        "geospatial_prior_reason": geo.reason,
        "text_evidence_score": _text_evidence_score(item, context),
        "comment_evidence_score": _comment_evidence_score(item, context),
        "is_target_positive": bucket in {"gold", "silver"},
        "is_negative_material": negative_reason is not None,
        "occurrence_bin": bucket,
        "bin_reason": reason,
    }


def _object_scoring_labels(candidate_set: CandidateSet) -> _ObjectScoringLabels:
    return _ObjectScoringLabels(
        family=tuple(_unique(candidate.family for candidate in candidate_set.family_candidates if candidate.family)),
        genus=tuple(
            _unique(
                candidate.genus
                for candidate in (*candidate_set.genus_candidates, *candidate_set.family_candidates)
                if candidate.genus
            )
        ),
        species=candidate_set.prompt_labels("species"),
    )


def _raise_if_hierarchical_classification(classification_mode: ClassificationMode) -> None:
    if classification_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION:
        raise ValueError(
            "target-scope object scoring helper received hierarchical_butterfly_classification; "
            "dispatch through hierarchical scoring with a taxonomy_store"
        )


def _validate_visual_top_k(
    *,
    family_top_k: int,
    species_first_pass_top_k: int,
    species_rerank_top_k: int,
) -> tuple[int, int, int]:
    family = int(family_top_k)
    first_pass = int(species_first_pass_top_k)
    rerank = int(species_rerank_top_k)
    if family <= 0:
        raise ValueError("family_top_k must be positive")
    if first_pass <= 0:
        raise ValueError("species_first_pass_top_k must be positive")
    if rerank <= 0:
        raise ValueError("species_rerank_top_k must be positive")
    if rerank > first_pass:
        raise ValueError("species_rerank_top_k must be <= species_first_pass_top_k")
    return family, first_pass, rerank


def _score_label_sets_for_items(
    scorer: ObjectBioClipScorer,
    items: Sequence[dict[str, Any]],
    label_sets: Mapping[str, Sequence[str]],
) -> dict[str, list[dict[str, float]]]:
    label_sets_by_name = {str(name): tuple(str(label) for label in labels) for name, labels in label_sets.items()}
    batch_scorer = getattr(scorer, "score_label_sets_batch", None)
    if callable(batch_scorer):
        return _coerce_label_set_batch_scores(batch_scorer(items, label_sets_by_name), label_sets_by_name, len(items))
    return {
        name: [scorer.score(item, labels) for item in items]
        for name, labels in label_sets_by_name.items()
    }


def _coerce_label_set_batch_scores(
    scores_by_label_set: Mapping[str, Sequence[Mapping[str, Any]]],
    label_sets: Mapping[str, Sequence[str]],
    expected_count: int,
) -> dict[str, list[dict[str, float]]]:
    output: dict[str, list[dict[str, float]]] = {}
    for name in label_sets:
        try:
            raw_scores = list(scores_by_label_set[name])
        except KeyError as exc:
            raise ValueError(f"BioCLIP batch scorer did not return label set {name!r}") from exc
        output[name] = _coerce_score_batch(raw_scores, expected_count=expected_count)
    return output


def _coerce_score_batch(scores_by_item: Sequence[Mapping[str, Any]], *, expected_count: int) -> list[dict[str, float]]:
    scores = list(scores_by_item)
    if len(scores) != expected_count:
        raise ValueError(f"BioCLIP batch scorer returned {len(scores)} rows for {expected_count} images")
    return [{str(label): float(score) for label, score in dict(row).items()} for row in scores]


def _rank_species(candidates: tuple[CandidateTaxon, ...], scores: dict[str, float]) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for candidate in candidates:
        labels = _candidate_species_labels(candidate)
        ranked.append((candidate.scientific_name, max(float(scores.get(label, 0.0)) for label in labels)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def _species_rerank_candidates(
    candidates: tuple[CandidateTaxon, ...],
    ranked_species_top20: list[tuple[str, float]],
) -> tuple[CandidateTaxon, ...]:
    candidates_by_name = {_norm(candidate.scientific_name): candidate for candidate in candidates}
    selected: list[CandidateTaxon] = []
    seen: set[str] = set()
    for name, _score in ranked_species_top20:
        key = _norm(name)
        candidate = candidates_by_name.get(key)
        if candidate is not None and key not in seen:
            selected.append(candidate)
            seen.add(key)
    return tuple(selected)


def _rank_species_for_family(
    *,
    candidates: tuple[CandidateTaxon, ...],
    ranked_species: list[tuple[str, float]],
    family: str | None,
) -> list[tuple[str, float]]:
    family_norm = _norm(family)
    if not ranked_species:
        return []
    if not family_norm:
        return ranked_species

    candidate_family_by_name: dict[str, str] = {}
    family_metadata_present = False
    for candidate in candidates:
        normalized_family = _norm(candidate.family)
        if normalized_family:
            family_metadata_present = True
        candidate_family_by_name[_norm(candidate.scientific_name)] = normalized_family

    if not family_metadata_present:
        return ranked_species

    filtered = [
        (name, score)
        for name, score in ranked_species
        if candidate_family_by_name.get(_norm(name)) == family_norm
    ]
    return filtered if filtered else ranked_species


def _target_scope_species_rerank_strategy(species_first_pass_top_k: int) -> str:
    return f"first_pass_top{int(species_first_pass_top_k)}"


def _species_prompt_labels(candidates: tuple[CandidateTaxon, ...]) -> tuple[str, ...]:
    return tuple(_unique(label for candidate in candidates for label in _candidate_species_labels(candidate)))


def _candidate_species_labels(candidate: CandidateTaxon) -> tuple[str, ...]:
    return (candidate.scientific_name, f"a photo of {candidate.scientific_name}", *candidate.common_names)


def _rank_labels(labels: tuple[str, ...], scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(((label, float(scores.get(label, 0.0))) for label in labels), key=lambda item: item[1], reverse=True)


def _taxon_key_by_name(candidates: tuple[CandidateTaxon, ...]) -> dict[str, str]:
    return {
        _norm(candidate.scientific_name): str(candidate.accepted_taxon_key)
        for candidate in candidates
        if candidate.accepted_taxon_key
    }


def _taxon_key_for_name(keys_by_name: dict[str, str], name: str | None) -> str | None:
    if not name:
        return None
    return keys_by_name.get(_norm(name))


def _candidate_for_name(candidates: tuple[CandidateTaxon, ...], name: str | None) -> CandidateTaxon | None:
    if not name:
        return None
    normalized = _norm(name)
    for candidate in candidates:
        if _norm(candidate.scientific_name) == normalized:
            return candidate
    return None


def _bucket(
    *,
    item: dict[str, Any],
    target_score: float,
    target_rank: int | None,
    margin: float | None,
    geo: GeospatialPrior,
) -> tuple[str, str]:
    from biominer.evidence.buckets import object_occurrence_bucket

    return object_occurrence_bucket(
        item=item,
        target_score=target_score,
        target_rank=target_rank,
        margin=margin,
        geo=geo,
    )


def _photo_summary(
    scores: pl.DataFrame,
    *,
    canonical: pl.DataFrame | None = None,
    detections: pl.DataFrame | None = None,
    species_context: SpeciesContext | None = None,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    summarized_keys: set[tuple[str, str]] = set()
    canonical_by_photo = _canonical_by_photo(canonical)
    detections_by_photo = _detections_by_photo(detections)
    if _has_columns(scores, ["source", "flickr_photo_id"]):
        for (_source, _photo), group in scores.group_by(["source", "flickr_photo_id"], maintain_order=True):
            sorted_rows = sorted(
                group.to_dicts(),
                key=_photo_summary_sort_key,
            )
            best = sorted_rows[0]
            key = (str(best["source"]), str(best["flickr_photo_id"]))
            detection_ids = _summary_detection_ids(detections_by_photo.get(key, []), sorted_rows)
            species = _summary_candidate_species(sorted_rows)
            hierarchical_fields = _hierarchical_photo_summary_fields(sorted_rows, best)
            photo_bucket, photo_reason = _photo_bucket_and_reason(sorted_rows, canonical_by_photo.get(key, {}))
            if hierarchical_fields["photo_multi_object_conflict"]:
                photo_bucket, photo_reason = "in_review", "multiple_species"
            hierarchical_fields["photo_review_reason"] = photo_reason if photo_bucket == "in_review" else ""
            summarized_keys.add(key)
            rows.append(
                {
                    "source": best["source"],
                    "flickr_photo_id": best["flickr_photo_id"],
                    "best_detection_id": best["detection_id"],
                    "detection_count": len(detection_ids),
                    "best_object_occurrence_bin": best["occurrence_bin"],
                    "best_object_species_top1": best["species_top1_scientific_name"],
                    "best_object_score": _summary_object_score(best),
                    "photo_occurrence_bin": photo_bucket,
                    "photo_bin_reason": photo_reason,
                    "all_detection_ids": detection_ids,
                    "all_candidate_species": species,
                    **hierarchical_fields,
                }
            )
    if canonical is not None:
        for record in canonical.to_dicts():
            key = (str(record.get("source") or ""), str(record.get("flickr_photo_id") or ""))
            if key in summarized_keys:
                continue
            fallback = _unscored_photo_summary(record, detections_by_photo.get(key, []), species_context)
            if fallback is not None:
                rows.append(fallback)
                summarized_keys.add(key)
    if not rows:
        return empty_photo_summary_frame()
    return _ensure_columns(pl.DataFrame(rows), PHOTO_EVIDENCE_SUMMARY_SCHEMA).select(
        list(PHOTO_EVIDENCE_SUMMARY_SCHEMA)
    )


def empty_photo_summary_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=PHOTO_EVIDENCE_SUMMARY_SCHEMA)


def _ensure_columns(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if frame.is_empty() and not frame.columns:
        return pl.DataFrame(schema=schema)
    expressions = [
        pl.col(name).cast(dtype).alias(name) if name in frame.columns else pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in schema.items()
    ]
    return frame.with_columns(expressions)


def _unscored_photo_summary(
    record: dict[str, Any],
    detection_rows: list[dict[str, Any]],
    species_context: SpeciesContext | None,
) -> dict[str, Any] | None:
    detections = [row for row in detection_rows if detection_is_bioclip_eligible(row)]
    detection_ids = _unique(row.get("detection_id") for row in detections)
    if detection_ids:
        return {
            "source": str(record.get("source") or ""),
            "flickr_photo_id": str(record.get("flickr_photo_id") or ""),
            "best_detection_id": detection_ids[0],
            "detection_count": len(detection_ids),
            "best_object_occurrence_bin": None,
            "best_object_species_top1": None,
            "best_object_score": None,
            "photo_occurrence_bin": "in_review",
            "photo_bin_reason": "detected_object_without_bioclip_score",
            "all_detection_ids": detection_ids,
            "all_candidate_species": [],
            **_empty_photo_prediction_fields(photo_review_reason="detected_object_without_bioclip_score"),
        }

    visual_negative_reason = _noneligible_detection_reason(detection_rows)
    if visual_negative_reason:
        return {
            "source": str(record.get("source") or ""),
            "flickr_photo_id": str(record.get("flickr_photo_id") or ""),
            "best_detection_id": None,
            "detection_count": 0,
            "best_object_occurrence_bin": None,
            "best_object_species_top1": None,
            "best_object_score": None,
            "photo_occurrence_bin": "bin",
            "photo_bin_reason": visual_negative_reason,
            "all_detection_ids": [],
            "all_candidate_species": [],
            **_empty_photo_prediction_fields(photo_review_reason=visual_negative_reason),
        }

    has_detection_failure = any(str(row.get("detection_status") or "") == "no_detection" for row in detection_rows)
    strong_text_evidence = species_context is not None and _text_evidence_score(record, species_context) > 0
    if not has_detection_failure and not strong_text_evidence:
        return None
    if has_detection_failure and not strong_text_evidence:
        failure_reason = _no_detection_failure_reason(detection_rows)
        return {
            "source": str(record.get("source") or ""),
            "flickr_photo_id": str(record.get("flickr_photo_id") or ""),
            "best_detection_id": None,
            "detection_count": 0,
            "best_object_occurrence_bin": None,
            "best_object_species_top1": None,
            "best_object_score": None,
            "photo_occurrence_bin": "bin" if failure_reason == "no_butterfly_like_object" else "in_review",
            "photo_bin_reason": failure_reason,
            "all_detection_ids": [],
            "all_candidate_species": [],
            **_empty_photo_prediction_fields(photo_review_reason=failure_reason),
        }
    review_reason = "no_detection_strong_text_evidence" if strong_text_evidence else "no_detection_without_object_score"
    return {
        "source": str(record.get("source") or ""),
        "flickr_photo_id": str(record.get("flickr_photo_id") or ""),
        "best_detection_id": None,
        "detection_count": 0,
        "best_object_occurrence_bin": None,
        "best_object_species_top1": None,
        "best_object_score": None,
        "photo_occurrence_bin": "in_review",
        "photo_bin_reason": review_reason,
        "all_detection_ids": [],
        "all_candidate_species": (
            [species_context.scientific_name] if strong_text_evidence and species_context is not None else []
        ),
        **_empty_photo_prediction_fields(photo_review_reason=review_reason),
    }


def _noneligible_detection_reason(detection_rows: list[dict[str, Any]]) -> str | None:
    labels = {
        str(row.get("detector_label") or "")
        for row in detection_rows
        if str(row.get("detection_status") or "") == "detected"
    }
    if not labels:
        return None
    if "hard_negative" in labels:
        return "negative_material_hard_negative_object"
    if labels <= {"moth_like", "insect_like"}:
        return "negative_material_non_target_order"
    if not any(label == "butterfly_like" for label in labels):
        return "negative_material_non_butterfly"
    return None


def _no_detection_failure_reason(detection_rows: list[dict[str, Any]]) -> str:
    for row in detection_rows:
        if str(row.get("detection_status") or "") != "no_detection":
            continue
        reason = str(row.get("failure_reason") or "").strip()
        if reason:
            return reason
    return "no_detection_without_object_score"


def _active_bioclip_gate_policy(
    *,
    detection_policy: DetectionPolicy | None,
    bioclip_gate_policy: BioClipGatePolicy | None,
    detected_visual_input_kind: AblationMode,
) -> BioClipGatePolicy:
    if bioclip_gate_policy is None:
        active_detection_policy = detection_policy or DetectionPolicy()
        bioclip_gate_policy = BioClipGatePolicy.legacy_butterfly_like_only(
            eligible_detector_labels=tuple(active_detection_policy.bioclip_eligible_labels)
        )
    return replace(bioclip_gate_policy, detected_visual_input_kind=detected_visual_input_kind)


def _score_detector_crop_decision(detection: dict[str, Any], *, gate_policy: BioClipGatePolicy) -> bool:
    decision = bioclip_score_input_decision(detection, gate_policy)
    return decision.should_score and decision.visual_input_kind == "detector_crop"


def _score_items_for_visual_input_kind(
    *,
    canonical_records_by_photo: dict[tuple[str, str], dict[str, Any]],
    detections: pl.DataFrame,
    gate_policy: BioClipGatePolicy,
    visual_input_kind: AblationMode,
) -> Iterator[dict[str, Any]]:
    for detection in detections.to_dicts():
        decision = bioclip_score_input_decision(detection, gate_policy)
        if not decision.should_score or decision.visual_input_kind != visual_input_kind:
            continue
        key = (str(detection.get("source") or ""), str(detection.get("flickr_photo_id") or ""))
        record = _canonical_record_for_detection(canonical_records_by_photo, key=key)
        crop_hash = str(detection.get("crop_hash") or "")
        visual_input_id = visual_input_id_for(
            source=str(detection.get("source") or ""),
            flickr_photo_id=str(detection.get("flickr_photo_id") or ""),
            detection_id=str(detection.get("detection_id") or ""),
            visual_input_kind=visual_input_kind,
            crop_hash=crop_hash,
        )
        yield {
            **detection,
            **record,
            **score_item_gate_fields(decision=decision, visual_input_id=visual_input_id, crop_hash=crop_hash),
            "ablation_mode": visual_input_kind,
        }


def _visual_input_id(item: dict[str, Any], *, ablation_mode: AblationMode) -> str:
    value = str(item.get("visual_input_id") or "")
    if value:
        return value
    crop_hash = str(item.get("crop_hash") or "")
    return visual_input_id_for(
        source=str(item.get("source") or ""),
        flickr_photo_id=str(item.get("flickr_photo_id") or ""),
        detection_id=str(item.get("detection_id") or ""),
        visual_input_kind=str(item.get("visual_input_kind") or ablation_mode),
        crop_hash=crop_hash,
    )


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _detections_by_photo(detections: pl.DataFrame | None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if detections is None or detections.is_empty():
        return {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in detections.to_dicts():
        key = (str(row.get("source") or ""), str(row.get("flickr_photo_id") or ""))
        grouped.setdefault(key, []).append(row)
    return grouped


def _canonical_by_photo(canonical: pl.DataFrame | None) -> dict[tuple[str, str], dict[str, Any]]:
    if canonical is None or canonical.is_empty():
        return {}
    return {
        (str(row.get("source") or ""), str(row.get("flickr_photo_id") or "")): row
        for row in canonical.to_dicts()
    }


def _summary_candidate_species(scored_rows: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for row in scored_rows:
        top1 = str(row.get("species_top1_scientific_name") or "")
        if top1:
            values.append(top1)
        for column in ("species_top5", "species_top20"):
            candidates = row.get(column) or []
            if isinstance(candidates, list | tuple):
                values.extend(str(value) for value in candidates if value)
    return _unique(values)


def _hierarchical_photo_summary_fields(sorted_rows: list[dict[str, Any]], best: dict[str, Any]) -> dict[str, Any]:
    selected_families = _unique(
        _selected_rank_name(row, "family")
        for row in sorted_rows
        if str(row.get("classification_mode") or "") == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    )
    selected_genera = _unique(
        _selected_rank_name(row, "genus")
        for row in sorted_rows
        if str(row.get("classification_mode") or "") == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    )
    top1_species = _summary_species_top1(best)
    species_identities = _unique(
        str(
            row.get("species_top1_accepted_taxon_key")
            or row.get("accepted_taxon_key")
            or _summary_species_top1(row)
        )
        for row in sorted_rows
        if str(row.get("classification_mode") or "") == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    )
    is_hierarchical = any(
        str(row.get("classification_mode") or "") == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
        for row in sorted_rows
    )
    return {
        "all_selected_families": selected_families,
        "all_selected_genera": selected_genera,
        "photo_selected_family": _selected_rank_name(best, "family") or None,
        "photo_selected_family_node_id": _selected_rank_node_id(best, "family") or None,
        "photo_selected_genus": _selected_rank_name(best, "genus") or None,
        "photo_selected_genus_node_id": _selected_rank_node_id(best, "genus") or None,
        "photo_species_top1": top1_species or None,
        "photo_species_top1_key": str(
            best.get("species_top1_accepted_taxon_key") or best.get("accepted_taxon_key") or ""
        )
        or None,
        "photo_species_confidence_score": _summary_object_score(best) if top1_species else None,
        "photo_species_margin": _summary_species_margin(best),
        "photo_multi_object_conflict": bool(
            is_hierarchical
            and (
                len(selected_families) > 1
                or len(selected_genera) > 1
                or len(species_identities) > 1
            )
        ),
        "photo_review_reason": "",
    }


def _empty_photo_prediction_fields(*, photo_review_reason: str = "") -> dict[str, Any]:
    return {
        "all_selected_families": [],
        "all_selected_genera": [],
        "photo_selected_family": None,
        "photo_selected_family_node_id": None,
        "photo_selected_genus": None,
        "photo_selected_genus_node_id": None,
        "photo_species_top1": None,
        "photo_species_top1_key": None,
        "photo_species_confidence_score": None,
        "photo_species_margin": None,
        "photo_multi_object_conflict": False,
        "photo_review_reason": photo_review_reason,
    }


def _selected_rank_name(row: dict[str, Any], prefix: str) -> str:
    selected = str(row.get(f"selected_{prefix}") or "")
    if str(row.get("classifier_schema_version") or "").startswith("butterfly-cascade-output-"):
        return selected
    return selected or str(row.get(f"{prefix}_top1") or "")


def _selected_rank_node_id(row: dict[str, Any], prefix: str) -> str:
    return str(row.get(f"selected_{prefix}_node_id") or "")


def _photo_summary_sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
    return (
        -_summary_object_score(row),
        -(_summary_species_margin(row) or 0.0),
        str(row.get("detection_id") or ""),
    )


def _summary_species_top1(row: dict[str, Any]) -> str:
    return str(row.get("species_top1_scientific_name") or row.get("species_top1") or "")


def _summary_species_margin(row: dict[str, Any]) -> float | None:
    margin = row.get("species_top1_margin")
    if margin not in (None, ""):
        return _optional_float(margin)
    margin = row.get("species_top1_top2_margin")
    if margin not in (None, ""):
        return _optional_float(margin)
    return None


def _summary_detection_ids(detection_rows: list[dict[str, Any]], scored_rows: list[dict[str, Any]]) -> list[str]:
    detection_ids = _unique(
        row.get("detection_id")
        for row in detection_rows
        if str(row.get("detection_status") or "") == "detected"
    )
    if detection_ids:
        return detection_ids
    return _unique(row.get("detection_id") for row in scored_rows)


def _summary_object_score(row: dict[str, Any]) -> float:
    if str(row.get("classification_mode") or "") == HIERARCHICAL_BUTTERFLY_CLASSIFICATION:
        species_score = row.get("species_top1_score")
        if species_score not in (None, ""):
            return float(species_score)
    target_score = row.get("target_species_score")
    if target_score not in (None, ""):
        return float(target_score)
    species_score = row.get("species_top1_score")
    if species_score not in (None, ""):
        return float(species_score)
    return 0.0


def _has_columns(frame: pl.DataFrame, columns: Iterable[str]) -> bool:
    existing = set(frame.columns)
    return all(column in existing for column in columns)


def _photo_bucket_and_reason(rows: list[dict[str, Any]], canonical_record: dict[str, Any]) -> tuple[str, str]:
    from biominer.evidence.buckets import photo_bucket_and_reason

    return photo_bucket_and_reason(rows, canonical_record)


def _hard_negative_photo_reason(record: dict[str, Any]) -> str | None:
    from biominer.evidence.buckets import object_hard_negative_reason

    return object_hard_negative_reason(record)


def _target_score(ranked: list[tuple[str, float]], target: str) -> float:
    for name, score in ranked:
        if _norm(name) == _norm(target):
            return score
    return 0.0


def _target_rank(ranked: list[tuple[str, float]], target: str) -> int | None:
    for index, (name, _score) in enumerate(ranked, start=1):
        if _norm(name) == _norm(target):
            return index
    return None


def _margin(ranked: list[tuple[str, float]]) -> float | None:
    if len(ranked) < 2:
        return None
    return ranked[0][1] - ranked[1][1]


def _text_evidence_score(item: dict[str, Any], context: SpeciesContext) -> float:
    text = " ".join(str(item.get(key) or "") for key in ("title", "description", "raw_tags", "tags"))
    return 1.0 if any(term.casefold() in text.casefold() for term in context.target_terms()) else 0.0


def _comment_evidence_score(item: dict[str, Any], context: SpeciesContext) -> float:
    text = str(item.get("comments_text") or "")
    return 1.0 if text and any(term.casefold() in text.casefold() for term in context.target_terms()) else 0.0


def _geo_prior_table_match(
    *,
    latitude: float,
    longitude: float,
    species_context: SpeciesContext,
    geo_prior_table: pl.DataFrame | None,
    candidate: CandidateTaxon | None = None,
) -> GeospatialPrior | None:
    if geo_prior_table is None or geo_prior_table.is_empty():
        return None
    for row in geo_prior_table.to_dicts():
        if not (
            _geo_prior_row_matches_context(row, species_context)
            or (candidate is not None and _geo_prior_row_matches_candidate(row, candidate))
        ):
            continue
        bbox = _geo_prior_bbox(row)
        if bbox and _coordinate_in_bbox(latitude=latitude, longitude=longitude, bbox=bbox):
            return GeospatialPrior(score=0.10, reason="within_geo_prior_table")
    return None


def _geo_prior_row_matches_context(row: dict[str, Any], species_context: SpeciesContext) -> bool:
    context_keys = {
        _norm(species_context.accepted_taxon_key),
        _norm(species_context.species_key),
    } - {""}
    for key_column in ("accepted_taxon_key", "target_accepted_taxon_key", "species_key", "accepted_usage_key"):
        value = _norm(row.get(key_column))
        if value and value in context_keys:
            return True

    context_names = {
        _norm(species_context.scientific_name),
        _norm(species_context.canonical_name),
    } - {""}
    for name_column in ("scientific_name", "accepted_scientific_name", "target_scientific_name", "canonical_name"):
        value = _norm(row.get(name_column))
        if value and value in context_names:
            return True
    return False


def _geo_prior_row_matches_candidate(row: dict[str, Any], candidate: CandidateTaxon) -> bool:
    candidate_keys = {
        _norm(candidate.accepted_taxon_key),
    } - {""}
    for key_column in ("accepted_taxon_key", "target_accepted_taxon_key", "species_key", "accepted_usage_key"):
        value = _norm(row.get(key_column))
        if value and value in candidate_keys:
            return True

    candidate_names = {
        _norm(candidate.scientific_name),
    } - {""}
    for name_column in ("scientific_name", "accepted_scientific_name", "target_scientific_name", "canonical_name"):
        value = _norm(row.get(name_column))
        if value and value in candidate_names:
            return True
    return False


def _geo_prior_bbox(row: dict[str, Any]) -> str | None:
    bbox = str(row.get("bbox") or "").strip()
    if bbox:
        return bbox
    parts = [_optional_float(row.get(column)) for column in ("min_lon", "min_lat", "max_lon", "max_lat")]
    if all(value is not None for value in parts):
        return ",".join(str(value) for value in parts)
    return None


def _coordinate_in_bbox(*, latitude: float, longitude: float, bbox: str) -> bool:
    min_lon, min_lat, max_lon, max_lat = (float(value) for value in bbox.split(","))
    return min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon


def _top_unique(values: Iterable[str], limit: int) -> list[str]:
    return _unique(value for value in values if value)[:limit]


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = str(value or "")
        key = _norm(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _ppm_bytes(data: bytes, *, width: int, height: int) -> bytes:
    expected = width * height * 3
    if len(data) != expected:
        raise ValueError(f"RGB crop bytes length {len(data)} does not match expected {expected}")
    return f"P6\n{width} {height}\n255\n".encode("ascii") + data


def _ablation_mode(item: dict[str, Any]) -> AblationMode:
    mode = str(item.get("ablation_mode") or "detector_crop")
    if mode not in set(OBJECT_VISUAL_MODES):
        raise ValueError(f"unsupported object BioCLIP ablation mode: {mode}")
    return mode  # type: ignore[return-value]


def _materialized_detector_crop_path(*, item: dict[str, Any], mode: AblationMode) -> Path | None:
    if "crop_path" not in item:
        return None
    path_value = item.get("crop_path")
    if path_value is None or str(path_value).strip() == "":
        raise ValueError("materialized detector crop item has blank crop_path")
    path = path_value if isinstance(path_value, Path) else Path(str(path_value))
    if not path.exists():
        raise FileNotFoundError(f"materialized detector crop path does not exist: {path}")
    return path


def _detector_crop_materialization_config(scorer: ObjectBioClipScorer) -> DetectorCropMaterializationConfig | None:
    config = getattr(scorer, "detector_crop_materialization_config", None)
    if not callable(config):
        return None
    value = config()
    if not isinstance(value, DetectorCropMaterializationConfig):
        raise TypeError("detector_crop_materialization_config must return DetectorCropMaterializationConfig")
    return value


def _scorer_supports_detector_crop_segmentation(scorer: ObjectBioClipScorer, item: dict[str, Any]) -> bool:
    supports = getattr(scorer, "supports_detector_crop_segmentation", None)
    if callable(supports):
        return bool(supports(item))
    return False


def _segmentation_status(*, mode: AblationMode, crops_scored: int, unavailable_count: int) -> str | None:
    if mode != "detector_crop_segmentation":
        return None
    if crops_scored and unavailable_count:
        return "partial"
    if crops_scored:
        return "available"
    if unavailable_count:
        return "unavailable"
    return "not_requested"


def _visual_mode_status(*, mode: AblationMode, crops_scored: int, unavailable_count: int) -> str:
    segmentation = _segmentation_status(mode=mode, crops_scored=crops_scored, unavailable_count=unavailable_count)
    if segmentation is not None:
        return segmentation
    return "available" if crops_scored else "no_scored_detections"


def _bytes_hash(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _safe_file_stem(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace("\\", "_")


def _cached_text_embedding_by_label(
    frame: pl.DataFrame,
    *,
    candidate_set_id: str,
    model_id: str,
    model_checkpoint: str,
) -> dict[str, list[float]]:
    if frame.is_empty():
        return {}
    filtered = frame.filter(
        (pl.col("candidate_set_id") == candidate_set_id)
        & (pl.col("model_id") == model_id)
        & (pl.col("model_checkpoint") == model_checkpoint)
    )
    return {str(row["label"]): _float_vector(row["embedding"]) for row in filtered.select(["label", "embedding"]).to_dicts()}


def _cached_image_embedding_by_crop_hash(
    frame: pl.DataFrame,
    *,
    model_id: str,
    model_checkpoint: str,
) -> dict[str, list[float]]:
    if frame.is_empty():
        return {}
    filtered = frame.filter((pl.col("model_id") == model_id) & (pl.col("model_checkpoint") == model_checkpoint))
    output: dict[str, list[float]] = {}
    for row in filtered.select(["crop_hash", "embedding"]).to_dicts():
        output.setdefault(str(row["crop_hash"]), _float_vector(row["embedding"]))
    return output


def _float_vector(value: Any) -> list[float]:
    return [float(item) for item in value]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"embedding dimensions differ: image={len(left)}, text={len(right)}")
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _string_value_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    filtered = frame.filter(pl.col(column).is_not_null() & (pl.col(column).cast(pl.String) != ""))
    if filtered.is_empty():
        return {}
    return {
        str(row[column]): int(row["count"])
        for row in filtered.group_by(column).len(name="count").sort(column).to_dicts()
    }


def _string_values(frame: pl.DataFrame, column: str) -> list[str]:
    if frame.is_empty() or column not in frame.columns:
        return []
    return sorted(
        {
            str(value)
            for value in frame.get_column(column).drop_nulls().to_list()
            if str(value or "").strip()
        }
    )


def _numeric_summary(frame: pl.DataFrame, column: str) -> dict[str, float | int | None]:
    empty = {
        f"{column}_non_null_count": 0,
        f"{column}_min": None,
        f"{column}_max": None,
        f"{column}_mean": None,
    }
    if frame.is_empty() or column not in frame.columns:
        return empty
    values = [float(value) for value in frame.get_column(column).drop_nulls().to_list()]
    if not values:
        return empty
    return {
        f"{column}_non_null_count": len(values),
        f"{column}_min": min(values),
        f"{column}_max": max(values),
        f"{column}_mean": sum(values) / len(values),
    }


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().split())
