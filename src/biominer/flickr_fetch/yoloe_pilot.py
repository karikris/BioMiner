"""Reproducible local YOLOE screening over a sampled Flickr image snapshot.

This is deliberately a coarse visual-screening tool.  It neither identifies
species nor turns Flickr discovery evidence into taxonomic validation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
import json
from math import sqrt
from pathlib import Path
import sqlite3
from typing import Any, Callable, Protocol, Sequence
from urllib.request import Request, urlopen

from PIL import Image
import polars as pl

from biominer.detection.detector_base import DecodedImage, DetectionCandidate
from biominer.detection.yoloe26_detector import YoloE26SidecarObjectDetector
from biominer.storage.parquet import write_parquet


PILOT_VERSION = "flickr-yoloe-screening-pilot-v1"
POSITIVE_LABELS = frozenset({"adult_butterfly", "possible_adult_butterfly", "moth_like", "caterpillar", "pupa", "insect_like"})


class YoloeBatchError(RuntimeError):
    """A model-wide batch failure that requires a fresh sidecar process."""


class Detector(Protocol):
    model_id: str
    model_version: str
    checkpoint: str
    prompt_classes: Sequence[str]
    prompt_set_fingerprint: str
    def detect_batch(self, images: Sequence[DecodedImage]) -> list[list[DetectionCandidate]]: ...
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class YoloePilotConfig:
    state_db: Path
    output_dir: Path
    reports_dir: Path
    sample_size: int = 10_000
    sample_seed: str = "australia-flickr-yoloe-10k-v1"
    runtime_python: Path = Path("../YOLO26/venv/bin/python")
    checkpoint: str = "yoloe-26s-seg.pt"
    device: str = "mps"
    imgsz: int = 768
    confidence: float = 0.20
    iou: float = 0.50
    max_det: int = 8
    download_workers: int = 4
    yoloe_batch_size: int = 8
    timeout_seconds: float = 30.0
    retries: int = 2
    max_image_bytes: int = 20_000_000
    max_attempts: int = 3
    prompt_classes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.sample_size <= 0 or self.download_workers <= 0 or self.yoloe_batch_size <= 0:
            raise ValueError("sample_size, download_workers, and yoloe_batch_size must be positive")
        if self.max_image_bytes <= 0 or self.max_attempts <= 0:
            raise ValueError("max_image_bytes and max_attempts must be positive")
        if self.device not in {"mps", "cpu", "auto"}:
            raise ValueError("device must be mps, cpu, or auto")
        if self.prompt_classes is not None and not self.prompt_classes:
            raise ValueError("prompt_classes must contain at least one prompt when provided")


@dataclass(frozen=True, slots=True)
class YoloePilotResult:
    sample_path: Path
    results_path: Path
    failures_path: Path
    report_path: Path
    summary_path: Path
    report: dict[str, Any]


def run_yoloe_pilot(config: YoloePilotConfig, *, detector: Detector | None = None) -> YoloePilotResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    sample_path = config.output_dir / "sample_register.parquet"
    sample = _sample_register(config, sample_path)
    state_path = config.output_dir / "pilot_state.sqlite"
    _init_state(state_path)
    owned_detector = detector is None
    detector = detector or YoloE26SidecarObjectDetector(
        runtime_python=str(config.runtime_python), checkpoint=config.checkpoint, device=config.device,
        imgsz=config.imgsz, conf=config.confidence, iou=config.iou, max_det=config.max_det,
        prompt_classes=config.prompt_classes,
    )
    try:
        _run_pending(sample, state_path, config, detector)
    finally:
        if owned_detector:
            detector.close()
    results, failures = _state_frames(state_path)
    results_path = write_parquet(results, config.output_dir / "image_level_routes.parquet")
    failures_path = write_parquet(failures, config.output_dir / "failures.parquet")
    report = _report(sample, results, failures, config, detector)
    report_path = config.reports_dir / f"yoloe_flickr_10k_{config.output_dir.name}.json"
    summary_path = report_path.with_suffix(".md")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(_markdown_summary(report), encoding="utf-8")
    return YoloePilotResult(sample_path, results_path, failures_path, report_path, summary_path, report)


def _sample_register(config: YoloePilotConfig, path: Path) -> pl.DataFrame:
    if path.exists():
        frame = pl.read_parquet(path)
        if frame.height != config.sample_size:
            raise ValueError("existing sample register has a different sample size")
        return frame
    with sqlite3.connect(config.state_db) as conn:
        rows = conn.execute("""
            SELECT source, flickr_photo_id, image_url, image_url_kind, source_record_hash,
                   query_term, query_language, query_field, created_at
            FROM source_records WHERE image_url != '' ORDER BY source, flickr_photo_id
        """).fetchall()
    if len(rows) < config.sample_size:
        raise ValueError(f"only {len(rows)} canonical records have an image URL; need {config.sample_size}")
    names = ("source", "flickr_photo_id", "image_url", "image_url_kind", "source_record_hash", "query_term", "query_language", "query_field", "source_created_at")
    sampled = sorted(rows, key=lambda row: sha256(f"{config.sample_seed}:{row[0]}:{row[1]}".encode()).hexdigest())[:config.sample_size]
    frame = (
        pl.DataFrame([dict(zip(names, row, strict=True)) for row in sampled])
        .with_row_index("sample_rank", offset=1)
        .with_columns(pl.lit(len(rows)).alias("snapshot_population_with_image_url"))
        .sort("sample_rank")
    )
    write_parquet(frame, path)
    return frame


def _init_state(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS results (
          flickr_photo_id TEXT PRIMARY KEY, result_json TEXT NOT NULL, completed_at TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS failures (
          flickr_photo_id TEXT PRIMARY KEY, attempts INTEGER NOT NULL, stage TEXT NOT NULL,
          error TEXT NOT NULL, updated_at TEXT NOT NULL)""")


def _run_pending(sample: pl.DataFrame, state_path: Path, config: YoloePilotConfig, detector: Detector) -> None:
    with sqlite3.connect(state_path) as conn:
        done = {item[0] for item in conn.execute("SELECT flickr_photo_id FROM results")}
        attempts = dict(conn.execute("SELECT flickr_photo_id, attempts FROM failures"))
    pending = [row for row in sample.iter_rows(named=True) if row["flickr_photo_id"] not in done and attempts.get(row["flickr_photo_id"], 0) < config.max_attempts]
    for start in range(0, len(pending), config.yoloe_batch_size):
        batch = pending[start:start + config.yoloe_batch_size]
        decoded: dict[str, DecodedImage] = {}
        with ThreadPoolExecutor(max_workers=config.download_workers) as pool:
            jobs = {pool.submit(_download_decode, row, config): row for row in batch}
            for job in as_completed(jobs):
                row = jobs[job]
                try:
                    decoded[str(row["flickr_photo_id"])] = job.result()
                except Exception as exc:  # operational failure is never a biological negative.
                    _record_failure(state_path, row, "download_or_decode", exc)
        rows = [row for row in batch if str(row["flickr_photo_id"]) in decoded]
        if not rows:
            continue
        try:
            detected = detector.detect_batch([decoded[str(row["flickr_photo_id"])] for row in rows])
            if len(detected) != len(rows):
                raise RuntimeError("YOLOE returned a different number of result rows")
            for row, candidates in zip(rows, detected, strict=True):
                _record_result(state_path, row, candidates)
        except Exception as exc:
            for row in rows:
                _record_failure(state_path, row, "yoloe", exc)
            raise YoloeBatchError(f"YOLOE batch failed: {type(exc).__name__}: {exc}") from exc


def _download_decode(row: dict[str, Any], config: YoloePilotConfig) -> DecodedImage:
    request = Request(str(row["image_url"]), headers={"User-Agent": "BioMiner/0.1 YOLOE pilot"})
    last: Exception | None = None
    for _ in range(config.retries + 1):
        try:
            with urlopen(request, timeout=config.timeout_seconds) as response:
                content_type = str(response.headers.get("Content-Type") or "")
                if not content_type.lower().startswith("image/"):
                    raise ValueError(f"unexpected content type {content_type!r}")
                payload = response.read(config.max_image_bytes + 1)
            if len(payload) > config.max_image_bytes:
                raise ValueError("image exceeds max_image_bytes")
            with Image.open(BytesIO(payload)) as image:
                rgb = image.convert("RGB")
                return DecodedImage(rgb.width, rgb.height, "RGB", rgb.tobytes(), source_uri=str(row["image_url"]))
        except Exception as exc:  # network and decoder failures stay retryable in the state store.
            last = exc
    raise OSError(f"image download/decode failed: {type(last).__name__}: {last}")


def _record_result(path: Path, row: dict[str, Any], detections: Sequence[DetectionCandidate]) -> None:
    labels = {item.label for item in detections}
    route = _route(labels)
    payload = {"flickr_photo_id": row["flickr_photo_id"], "source": row["source"], "sample_rank": row["sample_rank"], "route": route,
               "screening_positive": route in {"butterfly_like", "other_lepidoptera_or_life_stage", "other_insect_like"},
               "detection_count": len(detections), "labels": sorted(labels),
               "detections_json": json.dumps([{"label": item.label, "score": item.score, "bbox_xyxy": item.bbox_xyxy} for item in detections], separators=(",", ":"), sort_keys=True)}
    for field in ("shard_index", "global_sample_rank"):
        if field in row:
            payload[field] = row[field]
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT OR REPLACE INTO results VALUES (?, ?, ?)", (str(row["flickr_photo_id"]), json.dumps(payload, sort_keys=True), _now()))
        conn.execute("DELETE FROM failures WHERE flickr_photo_id = ?", (str(row["flickr_photo_id"]),))


def _record_failure(path: Path, row: dict[str, Any], stage: str, exc: Exception) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("""INSERT INTO failures VALUES (?, 1, ?, ?, ?)
          ON CONFLICT(flickr_photo_id) DO UPDATE SET attempts=attempts+1, stage=excluded.stage, error=excluded.error, updated_at=excluded.updated_at""",
          (str(row["flickr_photo_id"]), stage, f"{type(exc).__name__}: {exc}", _now()))


def _route(labels: set[str]) -> str:
    if labels & {"adult_butterfly", "possible_adult_butterfly"}: return "butterfly_like"
    if labels & {"moth_like", "caterpillar", "pupa"}: return "other_lepidoptera_or_life_stage"
    if "insect_like" in labels: return "other_insect_like"
    if labels & {"artifact", "pinned_specimen"}: return "artifact_or_specimen"
    if "no_relevant_organism" in labels: return "no_relevant_organism"
    return "no_relevant_detection"


def _state_frames(path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    with sqlite3.connect(path) as conn:
        results = [json.loads(row[0]) for row in conn.execute("SELECT result_json FROM results ORDER BY flickr_photo_id")]
        failures = [dict(zip(("flickr_photo_id", "attempts", "stage", "error", "updated_at"), row, strict=True)) for row in conn.execute("SELECT flickr_photo_id, attempts, stage, error, updated_at FROM failures ORDER BY flickr_photo_id")]
    return pl.DataFrame(results), pl.DataFrame(failures)


def _report(sample: pl.DataFrame, results: pl.DataFrame, failures: pl.DataFrame, config: YoloePilotConfig, detector: Detector) -> dict[str, Any]:
    counts = dict(results.group_by("route").len().iter_rows()) if not results.is_empty() else {}
    positives = sum(counts.get(item, 0) for item in ("butterfly_like", "other_lepidoptera_or_life_stage", "other_insect_like"))
    n = results.height
    low, high = _wilson(positives, n)
    return {"schema_version": PILOT_VERSION, "status": "complete" if n == sample.height else "incomplete_retryable", "generated_at": _now(),
            "scientific_scope": "YOLOE visual screening evidence only; not taxonomic validation or an actual-butterfly count.",
            "sample": {"population_with_image_url": int(sample["snapshot_population_with_image_url"][0]), "sample_size": sample.height, "seed": config.sample_seed, "sample_register": str(config.output_dir / "sample_register.parquet")},
            "runtime": {"model_id": detector.model_id, "model_version": detector.model_version, "checkpoint": detector.checkpoint, "prompt_classes": list(detector.prompt_classes), "prompt_set_fingerprint": detector.prompt_set_fingerprint, "device": config.device, "imgsz": config.imgsz, "confidence": config.confidence, "iou": config.iou, "max_det": config.max_det},
            "counts": {"classified": n, "operational_failures": failures.height, "routes": counts},
            "butterfly_or_insect_visual_screening_estimate": {"positive_images": positives, "classified_images": n, "proportion": positives / n if n else None, "wilson_95_interval": [low, high] if n else None}}


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0: return (0.0, 0.0)
    z = 1.959963984540054
    p = successes / total; denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * sqrt((p * (1-p) + z*z/(4*total)) / total) / denominator
    return (center-margin, center+margin)


def _markdown_summary(report: dict[str, Any]) -> str:
    estimate = report["butterfly_or_insect_visual_screening_estimate"]
    return "# YOLOE Flickr screening pilot\n\n" + report["scientific_scope"] + "\n\n" + f"Classified: {report['counts']['classified']}  \nOperational failures: {report['counts']['operational_failures']}  \nVisual insect/butterfly candidates: {estimate['positive_images']}\n"


def _now() -> str:
    return datetime.now(UTC).isoformat()
