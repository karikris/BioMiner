from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from PIL import Image
import polars as pl

from biominer.bioclip.bioclip import PersistentBioClipScorer
from biominer.bioclip.bioclip_worker import decoded_rgb_image_content_hash
from biominer.bioclip.model_registry import BioClipRuntime, ModelConfig
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.detection.detector_base import DecodedImage, DetectionCandidate
from biominer.detection.routing import route_detection
from biominer.detection.yoloe26_detector import YoloE26SidecarObjectDetector


PROTOTYPE_VISION_SMOKE_VERSION = "prototype-vision-smoke-v1.0.0"
PROTOTYPE_VISION_SMOKE_REPORT = "prototype_vision_smoke_report.json"
PROTOTYPE_VISION_SMOKE_SUMMARY = "prototype_vision_smoke_summary.md"
EXPECTED_IMAGE_COUNT = 5


@dataclass(frozen=True, slots=True)
class PrototypeVisionSmokeConfig:
    support_manifest: Path
    support_manifest_sha256: str
    readiness: Path
    readiness_sha256: str
    reference_media_ids: tuple[str, ...]
    output_dir: Path
    bioclip_runtime_python: Path
    bioclip_hf_cache_dir: Path
    yoloe_runtime_python: Path
    model_name: str
    model_revision: str
    open_clip_version: str
    yoloe_checkpoint: str = "yoloe-26s-seg.pt"
    device: str = "mps"
    yoloe_batch_size: int = 3
    bioclip_batch_size: int = 5
    yoloe_imgsz: int = 768
    yoloe_conf: float = 0.20
    yoloe_iou: float = 0.50
    yoloe_max_det: int = 8

    def __post_init__(self) -> None:
        for field in (
            "support_manifest",
            "readiness",
            "output_dir",
            "bioclip_runtime_python",
            "bioclip_hf_cache_dir",
            "yoloe_runtime_python",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)).expanduser())
        for field in (
            "support_manifest_sha256",
            "readiness_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if len(self.reference_media_ids) != EXPECTED_IMAGE_COUNT:
            raise ValueError("prototype vision smoke requires exactly five media IDs")
        if len(set(self.reference_media_ids)) != EXPECTED_IMAGE_COUNT:
            raise ValueError("prototype vision smoke media IDs must be unique")
        if self.device not in {"mps", "cpu"}:
            raise ValueError("prototype vision smoke device must be mps or cpu")
        if not 0 < self.yoloe_batch_size <= EXPECTED_IMAGE_COUNT:
            raise ValueError("yoloe_batch_size must be between one and five")
        if self.bioclip_batch_size != EXPECTED_IMAGE_COUNT:
            raise ValueError("bioclip_batch_size must be exactly five")

    @classmethod
    def read_json(cls, path: str | Path) -> PrototypeVisionSmokeConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("prototype vision smoke config must be an object")
        values = dict(payload)
        schema_version = values.pop("schema_version", None)
        if schema_version != PROTOTYPE_VISION_SMOKE_VERSION:
            raise ValueError("unsupported prototype vision smoke config schema")
        values["reference_media_ids"] = tuple(values["reference_media_ids"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class PrototypeVisionSmokeResult:
    report: dict[str, Any]
    report_path: Path
    summary_path: Path


def run_prototype_vision_smoke(
    config: PrototypeVisionSmokeConfig,
) -> PrototypeVisionSmokeResult:
    started = perf_counter()
    _verify_file(config.support_manifest, config.support_manifest_sha256)
    _verify_file(config.readiness, config.readiness_sha256)
    readiness = json.loads(config.readiness.read_text(encoding="utf-8"))
    if readiness.get("bank_status") != "prototype_only" or not readiness.get(
        "classification_authorised"
    ):
        raise ValueError("prototype readiness does not authorise classification")
    rows = _selected_support_rows(config)
    decoded, image_evidence = _decode_images(rows)

    bioclip_started = perf_counter()
    runtime = _bioclip_runtime(config)
    scorer = PersistentBioClipScorer(
        runtime=runtime,
        hf_cache_dir=config.bioclip_hf_cache_dir,
        device=config.device,
        image_resize_mode="longest",
        preprocess_workers=1,
    )
    try:
        scorer.ensure_model_attestation()
        embeddings = scorer.embed_image_paths(
            [Path(str(row["source_object_uri"])) for row in rows]
        )
        bioclip = _bioclip_evidence(scorer, embeddings, image_evidence, config)
    finally:
        scorer.close()
    bioclip_seconds = perf_counter() - bioclip_started

    yoloe_started = perf_counter()
    detector = YoloE26SidecarObjectDetector(
        runtime_python=str(
            _absolute_without_symlink_resolution(config.yoloe_runtime_python)
        ),
        checkpoint=config.yoloe_checkpoint,
        device=config.device,
        imgsz=config.yoloe_imgsz,
        conf=config.yoloe_conf,
        iou=config.yoloe_iou,
        max_det=config.yoloe_max_det,
        transport="json_b64",
    )
    detections: list[list[DetectionCandidate]] = []
    batch_sizes: list[int] = []
    try:
        for start in range(0, len(decoded), config.yoloe_batch_size):
            batch = decoded[start : start + config.yoloe_batch_size]
            batch_sizes.append(len(batch))
            detections.extend(detector.detect_batch(batch))
        yoloe = _yoloe_evidence(detector, detections, batch_sizes, config)
    finally:
        detector.close()
    yoloe_seconds = perf_counter() - yoloe_started

    per_image = []
    for row, evidence, embedding, candidates in zip(
        rows, image_evidence, embeddings, detections, strict=True
    ):
        route = _route(candidates)
        per_image.append(
            {
                "reference_media_id": row["reference_media_id"],
                "accepted_taxon_key": row["accepted_taxon_key"],
                "scientific_name": row["scientific_name"],
                "reference_group": row["reference_group"],
                **evidence,
                "embedding_sha256": _json_sha256(embedding),
                "detection_count": len(candidates),
                "route": route,
            }
        )
    report: dict[str, Any] = {
        "schema_version": PROTOTYPE_VISION_SMOKE_VERSION,
        "status": "passed",
        "image_count": EXPECTED_IMAGE_COUNT,
        "support_manifest_sha256": config.support_manifest_sha256,
        "readiness_sha256": config.readiness_sha256,
        "bank_status": readiness["bank_status"],
        "prototype_readiness_status": readiness["prototype_readiness_status"],
        "device_requested": config.device,
        "cpu_fallback_policy": {
            "pytorch_enable_mps_fallback": "1",
            "enabled_for_bioclip_sidecar": True,
            "enabled_for_yoloe_sidecar": True,
        },
        "batch_settings": {
            "bioclip": config.bioclip_batch_size,
            "yoloe": config.yoloe_batch_size,
            "yoloe_actual_batches": batch_sizes,
        },
        "bioclip": bioclip,
        "yoloe": yoloe,
        "per_image": per_image,
        "elapsed_seconds": {
            "bioclip": round(bioclip_seconds, 6),
            "yoloe": round(yoloe_seconds, 6),
            "total": round(perf_counter() - started, 6),
        },
        "semantics": {
            "model_output_is_taxonomic_validation": False,
            "prototype_only": True,
            "raw_scores_are_probabilities": False,
        },
    }
    report["report_fingerprint"] = canonical_semantic_fingerprint(report)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.output_dir / PROTOTYPE_VISION_SMOKE_REPORT
    summary_path = config.output_dir / PROTOTYPE_VISION_SMOKE_SUMMARY
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path.write_text(_summary(report), encoding="utf-8")
    return PrototypeVisionSmokeResult(report, report_path, summary_path)


def _selected_support_rows(
    config: PrototypeVisionSmokeConfig,
) -> list[dict[str, object]]:
    support = pl.read_parquet(config.support_manifest)
    by_id = {
        str(row["reference_media_id"]): row for row in support.iter_rows(named=True)
    }
    missing = [
        media_id for media_id in config.reference_media_ids if media_id not in by_id
    ]
    if missing:
        raise ValueError(f"smoke media IDs missing from support: {', '.join(missing)}")
    rows = [by_id[media_id] for media_id in config.reference_media_ids]
    if any(row["dataset_split"] != "support_train" for row in rows):
        raise ValueError("prototype smoke images must come from support_train")
    return rows


def _decode_images(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[DecodedImage], list[dict[str, object]]]:
    decoded = []
    evidence = []
    for row in rows:
        path = Path(str(row["source_object_uri"]))
        digest = _file_sha256(path)
        if digest != row["source_image_sha256"]:
            raise ValueError(f"support image hash mismatch: {path}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            decoded_content_sha256 = decoded_rgb_image_content_hash(rgb)
            decoded.append(
                DecodedImage(width, height, "RGB", rgb.tobytes(), source_uri=str(path))
            )
        evidence.append(
            {
                "image_sha256": digest,
                "decoded_width": width,
                "decoded_height": height,
                "decoded_mode": "RGB",
                "decoded_content_sha256": decoded_content_sha256,
            }
        )
    return decoded, evidence


def _bioclip_evidence(scorer, embeddings, image_evidence, config):  # noqa: ANN001
    dimensions = {len(row) for row in embeddings}
    if len(embeddings) != EXPECTED_IMAGE_COUNT or len(dimensions) != 1:
        raise RuntimeError("BioCLIP returned an invalid embedding shape")
    if not all(isfinite(value) for row in embeddings for value in row):
        raise RuntimeError("BioCLIP embeddings contain non-finite values")
    expected_hashes = [str(row["decoded_content_sha256"]) for row in image_evidence]
    if scorer.last_image_content_hashes != expected_hashes:
        raise RuntimeError("BioCLIP content hashes differ from support image hashes")
    metrics = scorer.cache_metrics
    if (
        metrics["bioclip_worker_process_starts"] != 1
        or metrics["bioclip_model_loads"] != 1
        or metrics["bioclip_model_cache_hits"] < 1
    ):
        raise RuntimeError("BioCLIP persistent model loading was not demonstrated")
    if scorer.device != config.device:
        raise RuntimeError(f"BioCLIP did not use requested device {config.device}")
    return {
        "model_id": scorer.model_id,
        "model_revision": scorer.model_revision,
        "model_weights_sha256": scorer.model_weights_sha256,
        "open_clip_version": scorer.open_clip_version,
        "open_clip_config_sha256": scorer.open_clip_config_sha256,
        "preprocessing_version": scorer.preprocessing_version,
        "preprocessing_config": scorer.preprocessing_config,
        "preprocessing_fingerprint": scorer.preprocessing_fingerprint,
        "image_resize_mode": scorer.effective_image_resize_mode,
        "device_actual": scorer.device,
        "gpu_name": scorer.gpu_name,
        "embedding_shape": [EXPECTED_IMAGE_COUNT, next(iter(dimensions))],
        "finite_values": True,
        "content_hashes_match": True,
        "embedding_batch_size": config.bioclip_batch_size,
        "persistent_loading": metrics,
        "memory": dict(scorer.memory_metrics),
    }


def _yoloe_evidence(detector, detections, batch_sizes, config):  # noqa: ANN001
    if len(detections) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError("YOLOE returned the wrong number of image results")
    if detector.worker_process_starts != 1 or detector.worker_request_count != len(
        batch_sizes
    ):
        raise RuntimeError("YOLOE persistent worker reuse was not demonstrated")
    return {
        "model_id": detector.model_id,
        "model_version": detector.model_version,
        "checkpoint": detector.checkpoint,
        "device_requested": config.device,
        "prompt_set_fingerprint": detector.prompt_set_fingerprint,
        "prompt_count": len(detector.prompt_classes),
        "persistent_worker_process_starts": detector.worker_process_starts,
        "persistent_worker_requests": detector.worker_request_count,
        "batch_sizes": batch_sizes,
        "detection_count": sum(len(rows) for rows in detections),
    }


def _route(candidates: Sequence[DetectionCandidate]) -> dict[str, object]:
    if candidates:
        winner = max(candidates, key=lambda item: item.score)
        row = {
            "detection_status": "detected",
            "detector_label": winner.label,
            "detector_score": winner.score,
            "detector_prompt": winner.detector_prompt,
        }
    else:
        row = {"detection_status": "no_detection"}
    return route_detection(row).as_row_fields()


def _bioclip_runtime(config: PrototypeVisionSmokeConfig) -> BioClipRuntime:
    model = ModelConfig(
        model_id="bioclip2_5_huge",
        display_name="BioCLIP 2.5 Huge",
        role="preferred",
        status="use_if_available",
        task="biology image-text classification and embedding",
        model_name=config.model_name,
        checkpoint=config.model_revision,
        package_name="open_clip_torch",
        package_version=config.open_clip_version,
        model_hash=f"hf-revision:{config.model_revision}",
    )
    return BioClipRuntime(
        model=model,
        home=config.bioclip_runtime_python.parent.parent,
        venv_python=_absolute_without_symlink_resolution(config.bioclip_runtime_python),
        package_version=config.open_clip_version,
        available=True,
    )


def _verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"artifact hash mismatch for {path}: {actual}")


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, field: str) -> None:
    text = str(value)
    if len(text) != 71 or not text.startswith("sha256:"):
        raise ValueError(f"{field} must be a SHA-256 fingerprint")
    try:
        int(text[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a SHA-256 fingerprint") from exc


def _summary(report: Mapping[str, object]) -> str:
    bioclip = report["bioclip"]
    yoloe = report["yoloe"]
    assert isinstance(bioclip, Mapping) and isinstance(yoloe, Mapping)
    return (
        "# Prototype five-image vision smoke\n\n"
        f"- Status: {report['status']}\n"
        f"- Device: {report['device_requested']}\n"
        f"- BioCLIP: {bioclip['model_id']} at {bioclip['model_revision']}\n"
        f"- Embedding shape: {bioclip['embedding_shape']}\n"
        f"- YOLOE: {yoloe['model_id']} ({yoloe['checkpoint']})\n"
        f"- Images: {report['image_count']}\n"
        "- Semantics: prototype screening evidence only\n"
    )


__all__ = [
    "EXPECTED_IMAGE_COUNT",
    "PROTOTYPE_VISION_SMOKE_REPORT",
    "PROTOTYPE_VISION_SMOKE_SUMMARY",
    "PROTOTYPE_VISION_SMOKE_VERSION",
    "PrototypeVisionSmokeConfig",
    "PrototypeVisionSmokeResult",
    "run_prototype_vision_smoke",
]
