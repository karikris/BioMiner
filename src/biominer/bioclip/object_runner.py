from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from pathlib import Path
import tempfile
from typing import Any, Iterable, Literal, Protocol

import polars as pl

from biominer.bioclip.candidate_sets import CandidateSet, CandidateTaxon
from biominer.bioclip.policy import DEFAULT_BUCKET_POLICY
from biominer.detection.cropper import crop_with_padding
from biominer.detection.detector_base import DecodedImage
from biominer.detection.segmentation import NoneSegmenter, Segmenter
from biominer.species.context import SpeciesContext
from biominer.storage.parquet import write_parquet


AblationMode = Literal["whole_image", "detector_crop", "detector_crop_segmentation"]


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


@dataclass(frozen=True)
class ObjectEvidenceOutputs:
    object_evidence_joined: Path
    photo_evidence_summary: Path


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

        segmented = self._segmenter.segment_crop(crop)
        if segmented is None:
            return crop.encoded_bytes, crop.crop_width, crop.crop_height, crop.crop_hash
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


def screen_object_detections(
    *,
    canonical_records: pl.DataFrame,
    detections: pl.DataFrame,
    species_context: SpeciesContext,
    candidate_set: CandidateSet,
    scorer: ObjectBioClipScorer,
    output_path: str | Path | None = None,
    ablation_mode: AblationMode = "detector_crop",
) -> ObjectScreenResult:
    records_by_photo = {
        (str(row.get("source") or ""), str(row.get("flickr_photo_id") or "")): row
        for row in canonical_records.to_dicts()
    }
    rows: list[dict[str, Any]] = []
    for detection in detections.to_dicts():
        if str(detection.get("detection_status") or "") != "detected":
            continue
        key = (str(detection.get("source") or ""), str(detection.get("flickr_photo_id") or ""))
        record = records_by_photo.get(key, {})
        item = {**record, **detection, "ablation_mode": ablation_mode}
        rows.append(_score_detection(item=item, context=species_context, candidate_set=candidate_set, scorer=scorer, ablation_mode=ablation_mode))
    frame = pl.DataFrame(rows) if rows else pl.DataFrame()
    output = Path(output_path) if output_path is not None else None
    if output is not None:
        write_parquet(frame, output)
    return ObjectScreenResult(
        frame=frame,
        output_path=output,
        records_seen=canonical_records.height,
        detections_seen=detections.height,
        crops_scored=len(rows),
    )


def apply_geospatial_soft_prior(
    row: dict[str, Any],
    species_context: SpeciesContext,
    *,
    visual_score: float,
    geo_prior_table: pl.DataFrame | None = None,
) -> GeospatialPrior:
    latitude = _optional_float(row.get("latitude"))
    longitude = _optional_float(row.get("longitude"))
    if latitude is None or longitude is None:
        return GeospatialPrior(score=0.0, reason="missing_geo")
    for region in species_context.regions:
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
) -> ObjectEvidenceOutputs:
    canonical = pl.read_parquet(canonical_records_path)
    detections = pl.read_parquet(detections_path)
    scores = pl.read_parquet(scores_path)
    joined = (
        scores.join(canonical, on=["source", "flickr_photo_id"], how="left", suffix="_canonical")
        .join(detections, on=["source", "flickr_photo_id", "detection_id", "crop_hash"], how="left", suffix="_detection")
    )
    summary = _photo_summary(scores)
    joined_path = write_parquet(joined, joined_output_path)
    summary_path = write_parquet(summary, photo_summary_output_path)
    return ObjectEvidenceOutputs(object_evidence_joined=joined_path, photo_evidence_summary=summary_path)


def _score_detection(
    *,
    item: dict[str, Any],
    context: SpeciesContext,
    candidate_set: CandidateSet,
    scorer: ObjectBioClipScorer,
    ablation_mode: AblationMode,
) -> dict[str, Any]:
    species_labels = candidate_set.prompt_labels("species")
    family_labels = tuple(candidate.family or "" for candidate in candidate_set.family_candidates)
    genus_labels = tuple(candidate.genus or "" for candidate in candidate_set.genus_candidates)
    species_scores = scorer.score(item, species_labels)
    ranked_species = _rank_species(candidate_set.species_candidates, species_scores)
    target_score = _target_score(ranked_species, context.scientific_name)
    top1_score = ranked_species[0][1] if ranked_species else 0.0
    margin = _margin(ranked_species)
    geo = apply_geospatial_soft_prior(item, context, visual_score=target_score)
    bucket, reason = _bucket(item=item, target_score=target_score, margin=margin, geo=geo)
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
        "family_top3": _top_unique(family_labels, 3),
        "family_top1": _top_unique(family_labels, 1)[0] if _top_unique(family_labels, 1) else None,
        "family_top1_score": top1_score,
        "family_margin": margin,
        "genus_top8": _top_unique(genus_labels, 8),
        "genus_top1": _top_unique(genus_labels, 1)[0] if _top_unique(genus_labels, 1) else None,
        "genus_top1_score": top1_score,
        "genus_margin": margin,
        "species_top20": [name for name, _score in ranked_species[:20]],
        "species_top5": [name for name, _score in ranked_species[:5]],
        "species_top1_scientific_name": ranked_species[0][0] if ranked_species else None,
        "species_top1_score": top1_score,
        "species_top1_margin": margin,
        "target_species_score": target_score,
        "target_species_rank": _target_rank(ranked_species, context.scientific_name),
        "geospatial_prior_score": geo.score,
        "geospatial_prior_reason": geo.reason,
        "text_evidence_score": _text_evidence_score(item, context),
        "comment_evidence_score": _comment_evidence_score(item, context),
        "is_target_positive": bucket in {"gold", "silver"},
        "is_negative_material": False,
        "occurrence_bin": bucket,
        "bin_reason": reason,
    }


def _rank_species(candidates: tuple[CandidateTaxon, ...], scores: dict[str, float]) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for candidate in candidates:
        labels = [candidate.scientific_name, f"a photo of {candidate.scientific_name}", *candidate.common_names]
        ranked.append((candidate.scientific_name, max(float(scores.get(label, 0.0)) for label in labels)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def _bucket(*, item: dict[str, Any], target_score: float, margin: float | None, geo: GeospatialPrior) -> tuple[str, str]:
    if geo.route_to_review:
        return "in_review", geo.reason
    if target_score >= DEFAULT_BUCKET_POLICY.gold_species_threshold and _has_geo(item) and _has_event_date(item):
        return "gold", "target_species_score_ge_070"
    if target_score >= DEFAULT_BUCKET_POLICY.silver_species_threshold:
        if not _has_geo(item):
            return "silver", "missing_geo"
        if not _has_event_date(item):
            return "silver", "missing_event_date"
        return "silver", "target_species_score_ge_035"
    return "bronze", "weak_species_score"


def _photo_summary(scores: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for (_source, _photo), group in scores.group_by(["source", "flickr_photo_id"], maintain_order=True):
        sorted_rows = group.sort("target_species_score", descending=True).to_dicts()
        best = sorted_rows[0]
        detection_ids = [str(row["detection_id"]) for row in sorted_rows]
        species = _unique(row["species_top1_scientific_name"] for row in sorted_rows if row.get("species_top1_scientific_name"))
        rows.append(
            {
                "source": best["source"],
                "flickr_photo_id": best["flickr_photo_id"],
                "best_detection_id": best["detection_id"],
                "detection_count": len(detection_ids),
                "best_object_occurrence_bin": best["occurrence_bin"],
                "best_object_species_top1": best["species_top1_scientific_name"],
                "best_object_score": best["target_species_score"],
                "photo_occurrence_bin": _photo_bucket([row["occurrence_bin"] for row in sorted_rows]),
                "photo_bin_reason": best["bin_reason"],
                "all_detection_ids": detection_ids,
                "all_candidate_species": species,
            }
        )
    return pl.DataFrame(rows)


def _photo_bucket(buckets: list[str]) -> str:
    for bucket in ("gold", "silver", "bronze", "in_review", "bin"):
        if bucket in buckets:
            return bucket
    return "in_review"


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


def _has_geo(row: dict[str, Any]) -> bool:
    return row.get("latitude") not in (None, "") and row.get("longitude") not in (None, "")


def _has_event_date(row: dict[str, Any]) -> bool:
    return bool(row.get("date_taken") or row.get("datetaken") or row.get("captured_at") or row.get("eventDate"))


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
    if mode not in {"whole_image", "detector_crop", "detector_crop_segmentation"}:
        raise ValueError(f"unsupported object BioCLIP ablation mode: {mode}")
    return mode  # type: ignore[return-value]


def _bytes_hash(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().split())
