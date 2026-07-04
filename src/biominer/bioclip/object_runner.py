from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from pathlib import Path
from shutil import rmtree
import tempfile
from typing import Any, Iterable, Literal, Protocol

import polars as pl

from biominer.bioclip.candidate_sets import CandidateSet, CandidateTaxon
from biominer.bioclip.policy import DEFAULT_BUCKET_POLICY
from biominer.detection.cropper import crop_with_padding
from biominer.detection.detector_base import DecodedImage
from biominer.detection.schema import DETECTION_OUTPUT_SCHEMA
from biominer.detection.segmentation import (
    NoneSegmenter,
    SegmentationUnavailable,
    Segmenter,
    detector_crop_mask_available,
    detector_masked_crop_bytes,
)
from biominer.species.context import SpeciesContext
from biominer.storage.parquet import write_parquet


PRIMARY_VISUAL_CLASSIFIER = "bioclip_object"
OBJECT_VISUAL_MODES: tuple[str, ...] = ("whole_image", "detector_crop", "detector_crop_segmentation")
AblationMode = Literal["whole_image", "detector_crop", "detector_crop_segmentation"]
OBJECT_SCORE_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "detection_id": pl.String,
    "crop_hash": pl.String,
    "model_id": pl.String,
    "model_version": pl.String,
    "model_checkpoint": pl.String,
    "candidate_set_id": pl.String,
    "classified_at": pl.String,
    "ablation_mode": pl.String,
    "triage_group_top": pl.String,
    "triage_group_scores": pl.Struct({"butterfly_like": pl.Float64}),
    "family_top3": pl.List(pl.String),
    "family_top1": pl.String,
    "family_top1_score": pl.Float64,
    "family_margin": pl.Float64,
    "genus_top8": pl.List(pl.String),
    "genus_top1": pl.String,
    "genus_top1_score": pl.Float64,
    "genus_margin": pl.Float64,
    "species_top20": pl.List(pl.String),
    "species_top20_accepted_taxon_keys": pl.List(pl.String),
    "species_top5": pl.List(pl.String),
    "species_top5_accepted_taxon_keys": pl.List(pl.String),
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
class MaterializedCropInputs:
    rows: list[dict[str, Any]]
    crop_path_by_hash: dict[str, Path]
    temp_dir: Path

    def cleanup(self) -> None:
        if self.temp_dir.exists():
            rmtree(self.temp_dir)


class FakeObjectBioClipScorer:
    model_id = "fake-bioclip"
    model_version = "test"
    model_checkpoint = "fake-checkpoint"

    def __init__(self, scores_by_crop: dict[str, dict[str, float]]) -> None:
        self.scores_by_crop = scores_by_crop

    def score(self, item: dict[str, Any], labels: tuple[str, ...]) -> dict[str, float]:
        scores = self.scores_by_crop.get(str(item.get("crop_hash") or ""), {})
        return {label: float(scores.get(label, 0.0)) for label in labels}


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
        image = self._image_loader(item)
        if not isinstance(image, DecodedImage):
            raise TypeError("image_loader must return a DecodedImage")
        mode = _ablation_mode(item)
        data, width, height, content_hash = self._visual_input_for_mode(item=item, image=image, mode=mode)
        crop_path = self._write_temp_ppm(data, width=width, height=height, crop_hash=f"{mode}:{content_hash}")
        try:
            return {str(label): float(score) for label, score in dict(self._scorer(crop_path, labels)).items()}
        finally:
            if not self._should_retain_debug_crop():
                crop_path.unlink(missing_ok=True)

    def supports_detector_crop_segmentation(self, item: dict[str, Any]) -> bool:
        return detector_crop_mask_available(item) or not isinstance(self._segmenter, NoneSegmenter)

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
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        safe_hash = crop_hash.replace(":", "_").replace("/", "_")
        if self._should_retain_debug_crop():
            path = self._temp_dir / f"{safe_hash}.ppm"
            path.write_bytes(_ppm_bytes(data, width=width, height=height))
            self._debug_crops_written += 1
            return path
        handle = tempfile.NamedTemporaryFile(prefix=f"{safe_hash}_", suffix=".ppm", dir=self._temp_dir, delete=False)
        try:
            handle.write(_ppm_bytes(data, width=width, height=height))
            return Path(handle.name)
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
    crop_padding_ratio: float = 0.12,
    crop_target_px: int = 336,
) -> MaterializedCropInputs:
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
            if str(detection.get("detection_status") or "") != "detected":
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


def screen_object_detections(
    *,
    canonical_records: pl.DataFrame,
    detections: pl.DataFrame,
    species_context: SpeciesContext,
    candidate_set: CandidateSet,
    scorer: ObjectBioClipScorer,
    output_path: str | Path | None = None,
    ablation_mode: AblationMode = "detector_crop",
    geo_prior_table: pl.DataFrame | None = None,
    parquet_batch_rows: int = 10000,
) -> ObjectScreenResult:
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
    try:
        for detection in detections.to_dicts():
            if str(detection.get("detection_status") or "") != "detected":
                continue
            key = (str(detection.get("source") or ""), str(detection.get("flickr_photo_id") or ""))
            record = _canonical_record_for_detection(records_by_photo, key=key)
            item = {**detection, **record, "ablation_mode": ablation_mode}
            if ablation_mode == "detector_crop_segmentation" and not _scorer_supports_detector_crop_segmentation(scorer, item):
                segmentation_unavailable_count += 1
                segmentation_unavailable_reason = segmentation_unavailable_reason or "detector_masks_missing"
                continue
            try:
                score_row = _score_detection(
                    item=item,
                    context=species_context,
                    candidate_set=candidate_set,
                    scorer=scorer,
                    ablation_mode=ablation_mode,
                    geo_prior_table=geo_prior_table,
                )
            except SegmentationUnavailable as exc:
                if ablation_mode != "detector_crop_segmentation":
                    raise
                segmentation_unavailable_count += 1
                segmentation_unavailable_reason = segmentation_unavailable_reason or str(exc) or "detector_masks_missing"
                continue
            crops_scored += 1
            if output is None or batch_dir is None:
                rows.append(score_row)
            else:
                _buffer_score_rows(
                    [score_row],
                    row_buffer=row_buffer,
                    batch_paths=batch_paths,
                    batch_dir=batch_dir,
                    parquet_batch_rows=parquet_batch_rows,
                )
        if output is not None and batch_dir is not None:
            _flush_score_row_buffer(row_buffer=row_buffer, batch_paths=batch_paths, batch_dir=batch_dir)
            frame = _read_score_batches(batch_paths)
            write_parquet(frame, output)
        else:
            frame = pl.DataFrame(rows) if rows else empty_object_score_frame()
        return ObjectScreenResult(
            frame=frame,
            output_path=output,
            records_seen=canonical_records.height,
            detections_seen=detections.height,
            crops_scored=crops_scored,
            score_batches_written=len(batch_paths),
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
    finally:
        if batch_dir is not None and batch_dir.exists():
            rmtree(batch_dir)


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


def _read_score_batches(batch_paths: list[Path]) -> pl.DataFrame:
    if not batch_paths:
        return empty_object_score_frame()
    frames = [pl.read_parquet(path) for path in batch_paths]
    return pl.concat(frames, how="diagonal_relaxed")


def empty_object_score_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=OBJECT_SCORE_OUTPUT_SCHEMA)


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
    join_keys = ["source", "flickr_photo_id", "detection_id", "crop_hash"]
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
) -> dict[str, Any]:
    species_labels = candidate_set.prompt_labels("species")
    family_labels = tuple(_unique(candidate.family for candidate in candidate_set.family_candidates if candidate.family))
    genus_labels = tuple(
        _unique(
            candidate.genus
            for candidate in (*candidate_set.genus_candidates, *candidate_set.family_candidates)
            if candidate.genus
        )
    )
    family_scores = scorer.score(item, family_labels) if family_labels else {}
    genus_scores = scorer.score(item, genus_labels) if genus_labels else {}
    species_scores = scorer.score(item, species_labels)
    ranked_families = _rank_labels(family_labels, family_scores)
    ranked_genera = _rank_labels(genus_labels, genus_scores)
    ranked_species_top20 = _rank_species(candidate_set.species_candidates, species_scores)[:20]
    rerank_candidates = _species_rerank_candidates(
        candidate_set.species_candidates,
        ranked_species_top20,
        target_scientific_name=context.scientific_name,
    )
    rerank_scores = scorer.score(item, _species_prompt_labels(rerank_candidates)) if rerank_candidates else {}
    ranked_species = _rank_species(rerank_candidates, rerank_scores) if rerank_candidates else ranked_species_top20
    target_score = _target_score(ranked_species, context.scientific_name)
    top1_name = ranked_species[0][0] if ranked_species else None
    species_top20 = [name for name, _score in ranked_species_top20]
    species_top5 = [name for name, _score in ranked_species[:5]]
    taxon_key_by_name = _taxon_key_by_name(candidate_set.species_candidates)
    top1_taxon_key = _taxon_key_for_name(taxon_key_by_name, top1_name)
    top1_candidate = _candidate_for_name(candidate_set.species_candidates, top1_name)
    top1_score = ranked_species[0][1] if ranked_species else 0.0
    target_rank = _target_rank(ranked_species, context.scientific_name)
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
        "model_id": scorer.model_id,
        "model_version": scorer.model_version,
        "model_checkpoint": scorer.model_checkpoint,
        "candidate_set_id": candidate_set.candidate_set_id,
        "classified_at": datetime.now(UTC).isoformat(),
        "ablation_mode": ablation_mode,
        "triage_group_top": "butterfly_like",
        "triage_group_scores": {"butterfly_like": float(item.get("detector_score") or 0.0)},
        "family_top3": [name for name, _score in ranked_families[:3]],
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


def _rank_species(candidates: tuple[CandidateTaxon, ...], scores: dict[str, float]) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for candidate in candidates:
        labels = _candidate_species_labels(candidate)
        ranked.append((candidate.scientific_name, max(float(scores.get(label, 0.0)) for label in labels)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def _species_rerank_candidates(
    candidates: tuple[CandidateTaxon, ...],
    ranked_species_top20: list[tuple[str, float]],
    *,
    target_scientific_name: str,
) -> tuple[CandidateTaxon, ...]:
    candidates_by_name = {_norm(candidate.scientific_name): candidate for candidate in candidates}
    selected: list[CandidateTaxon] = []
    for name, _score in ranked_species_top20[:5]:
        candidate = candidates_by_name.get(_norm(name))
        if candidate is not None:
            selected.append(candidate)
    target = candidates_by_name.get(_norm(target_scientific_name))
    if target is not None and all(_norm(candidate.scientific_name) != _norm(target.scientific_name) for candidate in selected):
        selected.append(target)
    return tuple(selected)


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
    if _has_columns(scores, ["source", "flickr_photo_id", "target_species_score"]):
        for (_source, _photo), group in scores.group_by(["source", "flickr_photo_id"], maintain_order=True):
            sorted_rows = group.sort("target_species_score", descending=True).to_dicts()
            best = sorted_rows[0]
            key = (str(best["source"]), str(best["flickr_photo_id"]))
            detection_ids = _summary_detection_ids(detections_by_photo.get(key, []), sorted_rows)
            species = _summary_candidate_species(sorted_rows)
            photo_bucket, photo_reason = _photo_bucket_and_reason(sorted_rows, canonical_by_photo.get(key, {}))
            summarized_keys.add(key)
            rows.append(
                {
                    "source": best["source"],
                    "flickr_photo_id": best["flickr_photo_id"],
                    "best_detection_id": best["detection_id"],
                    "detection_count": len(detection_ids),
                    "best_object_occurrence_bin": best["occurrence_bin"],
                    "best_object_species_top1": best["species_top1_scientific_name"],
                    "best_object_score": best["target_species_score"],
                    "photo_occurrence_bin": photo_bucket,
                    "photo_bin_reason": photo_reason,
                    "all_detection_ids": detection_ids,
                    "all_candidate_species": species,
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
    return pl.DataFrame(rows) if rows else empty_photo_summary_frame()


def empty_photo_summary_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=PHOTO_EVIDENCE_SUMMARY_SCHEMA)


def _ensure_columns(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if frame.is_empty() and not frame.columns:
        return pl.DataFrame(schema=schema)
    missing = [pl.lit(None, dtype=dtype).alias(name) for name, dtype in schema.items() if name not in frame.columns]
    return frame.with_columns(missing) if missing else frame


def _unscored_photo_summary(
    record: dict[str, Any],
    detection_rows: list[dict[str, Any]],
    species_context: SpeciesContext | None,
) -> dict[str, Any] | None:
    detections = [row for row in detection_rows if str(row.get("detection_status") or "") == "detected"]
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
        }
    return {
        "source": str(record.get("source") or ""),
        "flickr_photo_id": str(record.get("flickr_photo_id") or ""),
        "best_detection_id": None,
        "detection_count": 0,
        "best_object_occurrence_bin": None,
        "best_object_species_top1": None,
        "best_object_score": None,
        "photo_occurrence_bin": "in_review",
        "photo_bin_reason": "no_detection_strong_text_evidence" if strong_text_evidence else "no_detection_without_object_score",
        "all_detection_ids": [],
        "all_candidate_species": [species_context.scientific_name] if strong_text_evidence and species_context is not None else [],
    }


def _no_detection_failure_reason(detection_rows: list[dict[str, Any]]) -> str:
    for row in detection_rows:
        if str(row.get("detection_status") or "") != "no_detection":
            continue
        reason = str(row.get("failure_reason") or "").strip()
        if reason:
            return reason
    return "no_detection_without_object_score"


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


def _summary_detection_ids(detection_rows: list[dict[str, Any]], scored_rows: list[dict[str, Any]]) -> list[str]:
    detection_ids = _unique(
        row.get("detection_id")
        for row in detection_rows
        if str(row.get("detection_status") or "") == "detected"
    )
    if detection_ids:
        return detection_ids
    return _unique(row.get("detection_id") for row in scored_rows)


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


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().split())
