from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from shutil import rmtree
from typing import Any

import polars as pl

from biominer.detection.cropper import crop_with_padding
from biominer.detection.detector_base import DecodedImage
from biominer.vision.gates import BioClipGatePolicy, ScoreInputDecision, bioclip_score_input_decision


BIOCLIP_SCORE_INPUT_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "detection_id": pl.String,
    "visual_input_id": pl.String,
    "visual_input_kind": pl.String,
    "visual_input_path": pl.String,
    "crop_hash": pl.String,
    "detector_label": pl.String,
    "detection_status": pl.String,
    "bioclip_gate_mode": pl.String,
    "bioclip_gate_decision": pl.String,
    "bioclip_gate_reason": pl.String,
    "detection_route": pl.String,
    "routing_action": pl.String,
    "bioclip_route": pl.String,
    "routing_priority": pl.String,
    "routing_reason": pl.String,
    "routing_policy_version": pl.String,
    "routing_policy_fingerprint": pl.String,
    "batch_id": pl.String,
    "part_id": pl.String,
}


@dataclass(frozen=True)
class MaterializedBioClipScoreInputs:
    frame: pl.DataFrame
    items: list[dict[str, Any]]
    temp_dir: Path

    @property
    def paths(self) -> list[Path]:
        return [Path(str(row["visual_input_path"])) for row in self.frame.to_dicts()]

    def cleanup(self) -> None:
        if self.temp_dir.exists():
            rmtree(self.temp_dir)


def materialize_bioclip_score_inputs(
    *,
    canonical_records: pl.DataFrame,
    detections: pl.DataFrame,
    image_loader: Any,
    temp_dir: str | Path,
    gate_policy: BioClipGatePolicy | None = None,
    crop_padding_ratio: float = 0.12,
    crop_target_px: int = 336,
    batch_id: str = "",
    part_id: str = "",
) -> MaterializedBioClipScoreInputs:
    active_gate_policy = gate_policy or BioClipGatePolicy()
    records_by_photo = {
        (str(row.get("source") or ""), str(row.get("flickr_photo_id") or "")): row
        for row in canonical_records.to_dicts()
    }
    root = Path(temp_dir)
    root.mkdir(parents=True, exist_ok=True)
    materialized_dir = root / f".bioclip_score_inputs_{_safe_file_stem(batch_id or part_id or 'batch')}"
    if materialized_dir.exists():
        rmtree(materialized_dir)
    materialized_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    image_by_photo: dict[tuple[str, str], DecodedImage] = {}
    try:
        for detection in detections.to_dicts():
            decision = bioclip_score_input_decision(detection, active_gate_policy)
            if not decision.should_score:
                continue
            key = (str(detection.get("source") or ""), str(detection.get("flickr_photo_id") or ""))
            record = _canonical_record_for_detection(records_by_photo, key=key)
            image = image_by_photo.get(key)
            if image is None:
                loaded = image_loader({**detection, **record})
                if not isinstance(loaded, DecodedImage):
                    raise TypeError("image_loader must return a DecodedImage")
                image_by_photo[key] = loaded
                image = loaded
            if decision.visual_input_kind == "detector_crop":
                data, width, height, crop_hash = _detector_crop_bytes(
                    detection,
                    image=image,
                    crop_padding_ratio=crop_padding_ratio,
                    crop_target_px=crop_target_px,
                )
            elif decision.visual_input_kind == "whole_image":
                data, width, height, crop_hash = image.data, image.width, image.height, _bytes_hash(image.data)
            else:
                continue
            visual_input_id = visual_input_id_for(
                source=str(detection.get("source") or ""),
                flickr_photo_id=str(detection.get("flickr_photo_id") or ""),
                detection_id=str(detection.get("detection_id") or ""),
                visual_input_kind=str(decision.visual_input_kind),
                crop_hash=crop_hash,
            )
            path = materialized_dir / f"{len(rows):06d}_{_safe_file_stem(visual_input_id)}.ppm"
            path.write_bytes(_ppm_bytes(data, width=width, height=height))
            row = {
                "source": str(detection.get("source") or ""),
                "flickr_photo_id": str(detection.get("flickr_photo_id") or ""),
                "detection_id": str(detection.get("detection_id") or ""),
                "visual_input_id": visual_input_id,
                "visual_input_kind": str(decision.visual_input_kind),
                "visual_input_path": str(path),
                "crop_hash": crop_hash,
                "detector_label": str(detection.get("detector_label") or ""),
                "detection_status": str(detection.get("detection_status") or ""),
                **decision.as_row_fields(),
                "batch_id": str(batch_id or ""),
                "part_id": str(part_id or ""),
            }
            rows.append(row)
            items.append(
                {
                    **detection,
                    **record,
                    **_score_item_gate_fields(decision=decision, visual_input_id=visual_input_id, crop_hash=crop_hash),
                    "ablation_mode": str(decision.visual_input_kind),
                    "crop_hash": crop_hash,
                    "crop_path": path,
                    "visual_input_path": str(path),
                }
            )
    except Exception:
        if materialized_dir.exists():
            rmtree(materialized_dir)
        raise
    return MaterializedBioClipScoreInputs(
        frame=_ensure_score_input_schema(pl.DataFrame(rows)),
        items=items,
        temp_dir=materialized_dir,
    )


def visual_input_id_for(
    *,
    source: str,
    flickr_photo_id: str,
    detection_id: str,
    visual_input_kind: str,
    crop_hash: str,
) -> str:
    payload = {
        "source": source,
        "flickr_photo_id": flickr_photo_id,
        "detection_id": detection_id,
        "visual_input_kind": visual_input_kind,
        "crop_hash": crop_hash,
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def score_item_gate_fields(
    *,
    decision: ScoreInputDecision,
    visual_input_id: str,
    crop_hash: str,
) -> dict[str, str | None]:
    return _score_item_gate_fields(decision=decision, visual_input_id=visual_input_id, crop_hash=crop_hash)


def _score_item_gate_fields(
    *,
    decision: ScoreInputDecision,
    visual_input_id: str,
    crop_hash: str,
) -> dict[str, str | None]:
    return {
        "visual_input_id": visual_input_id,
        **decision.as_row_fields(),
        "crop_hash": crop_hash,
    }


def _detector_crop_bytes(
    detection: dict[str, Any],
    *,
    image: DecodedImage,
    crop_padding_ratio: float,
    crop_target_px: int,
) -> tuple[bytes, int, int, str]:
    bbox = detection.get("bbox_xyxy")
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        raise ValueError("detector crop score input requires bbox_xyxy")
    crop = crop_with_padding(
        image,
        bbox_xyxy=tuple(float(value) for value in bbox),  # type: ignore[arg-type]
        padding_ratio=crop_padding_ratio,
        target_px=crop_target_px,
    )
    return crop.encoded_bytes, crop.crop_width, crop.crop_height, str(detection.get("crop_hash") or crop.crop_hash)


def _canonical_record_for_detection(
    records_by_photo: dict[tuple[str, str], dict[str, Any]],
    *,
    key: tuple[str, str],
) -> dict[str, Any]:
    record = records_by_photo.get(key)
    if record is None:
        source, photo_id = key
        raise ValueError(f"BioCLIP score input has no canonical source record: source={source!r}, flickr_photo_id={photo_id!r}")
    return record


def _ensure_score_input_schema(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() and not frame.columns:
        return pl.DataFrame(schema=BIOCLIP_SCORE_INPUT_SCHEMA)
    expressions = [
        pl.col(name).cast(dtype).alias(name) if name in frame.columns else pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in BIOCLIP_SCORE_INPUT_SCHEMA.items()
    ]
    return frame.with_columns(expressions)


def _ppm_bytes(data: bytes, *, width: int, height: int) -> bytes:
    expected = width * height * 3
    if len(data) != expected:
        raise ValueError(f"RGB score input bytes length {len(data)} does not match expected {expected}")
    return f"P6\n{width} {height}\n255\n".encode("ascii") + data


def _bytes_hash(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _safe_file_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:96] or "score_input"


__all__ = [
    "BIOCLIP_SCORE_INPUT_SCHEMA",
    "MaterializedBioClipScoreInputs",
    "materialize_bioclip_score_inputs",
    "score_item_gate_fields",
    "visual_input_id_for",
]
