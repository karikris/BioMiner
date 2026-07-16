"""Resumable local staged Flickr inference for the Phase 14 prototype."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from datetime import UTC, datetime
import hashlib
import json
from math import isfinite, sqrt
import os
from pathlib import Path
import resource
import sqlite3
import sys
from time import perf_counter, sleep
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image
import polars as pl

from biominer.bioclip.bioclip import PersistentBioClipScorer
from biominer.bioclip.bioclip_worker import decoded_rgb_image_content_hash
from biominer.bioclip.model_registry import BioClipRuntime, ModelConfig
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.detection.detector_base import DecodedImage, DetectionCandidate
from biominer.detection.routing import route_detection
from biominer.detection.yoloe26_detector import YoloE26SidecarObjectDetector


STAGED_FLICKR_VERSION = "prototype-staged-flickr-classification-v1.0.0"
STAGED_FLICKR_REPORT_VERSION = "prototype-staged-flickr-report-v1.0.0"
STAGED_FLICKR_RESULT_VERSION = "prototype-staged-flickr-results-v1.0.0"
STAGED_FLICKR_CANDIDATE_VERSION = "prototype-staged-flickr-candidates-v1.0.0"
STAGED_FLICKR_FAILURE_VERSION = "prototype-staged-flickr-failures-v1.0.0"
STAGED_FLICKR_RESULTS_FILE = "prototype_flickr_classification.parquet"
STAGED_FLICKR_CANDIDATES_FILE = "prototype_flickr_candidate_scores.parquet"
STAGED_FLICKR_FAILURES_FILE = "prototype_flickr_failures.parquet"
STAGED_FLICKR_REPORT_FILE = "prototype_staged_flickr_report.json"
STAGED_FLICKR_SUMMARY_FILE = "prototype_staged_flickr_summary.md"
STAGED_FLICKR_PROGRESS_FILE = "prototype_staged_flickr_progress.json"
STAGED_FLICKR_STATE_FILE = "prototype_staged_flickr_state.sqlite"
RAW_FULL_IMAGE = "raw_full_image"
SCORE_SEMANTICS = "experimental_screening_evidence_uncalibrated_not_probability"

_AUXILIARY_TEXT_CLASSES: tuple[tuple[str, str, str], ...] = (
    ("known_negative", "known-negative:moth", "a field photograph of a moth"),
    (
        "known_negative",
        "known-negative:other-insect",
        "a field photograph of an insect that is not a butterfly",
    ),
    ("visual_domain", "visual-domain:artwork", "butterfly artwork"),
    ("visual_domain", "visual-domain:logo", "a butterfly logo"),
    ("visual_domain", "visual-domain:tattoo", "a butterfly tattoo"),
    (
        "visual_domain",
        "visual-domain:pinned-specimen",
        "a pinned butterfly specimen",
    ),
    (
        "visual_domain",
        "visual-domain:partial-wing",
        "a photograph of only part of a butterfly wing",
    ),
    (
        "visual_domain",
        "visual-domain:dead-specimen",
        "a dead butterfly specimen",
    ),
    ("visual_domain", "visual-domain:flower", "a close photograph of a flower"),
    ("visual_domain", "visual-domain:fruit", "a close photograph of fruit"),
    ("visual_domain", "visual-domain:garden", "a wide garden scene"),
    ("visual_domain", "visual-domain:clutter", "a cluttered scene of objects"),
    (
        "visual_domain",
        "visual-domain:printed-image",
        "a printed image of a butterfly",
    ),
)


@dataclass(frozen=True, slots=True)
class PrototypeStagedFlickrConfig:
    geography: Path
    geography_sha256: str
    assignments: Path
    assignments_sha256: str
    query_hits: Path
    query_hits_sha256: str
    regional_competitors: Path
    regional_competitors_sha256: str
    readiness: Path
    readiness_sha256: str
    reference_embeddings: Path
    reference_embeddings_sha256: str
    reference_prototypes: Path
    reference_prototypes_sha256: str
    output_dir: Path
    bioclip_runtime_python: Path
    bioclip_hf_cache_dir: Path
    yoloe_runtime_python: Path
    model_name: str
    model_revision: str
    open_clip_version: str
    stage_limits: tuple[int, ...]
    target_accepted_taxon_key: str
    target_scientific_name: str
    storage_backend: str = "local"
    s3_permitted: bool = False
    device: str = "mps"
    yoloe_checkpoint: str = "yoloe-26s-seg.pt"
    download_workers: int = 4
    bioclip_batch_size: int = 16
    yoloe_batch_size: int = 8
    preprocess_workers: int = 4
    yoloe_imgsz: int = 768
    yoloe_conf: float = 0.20
    yoloe_iou: float = 0.50
    yoloe_max_det: int = 8
    max_failure_rate: float = 0.20
    max_record_attempts: int = 3
    request_timeout_seconds: float = 30.0
    request_retries: int = 2
    max_image_bytes: int = 20_000_000
    resume: bool = True
    retry_failed: bool = True

    def __post_init__(self) -> None:
        for field in (
            "geography",
            "assignments",
            "query_hits",
            "regional_competitors",
            "readiness",
            "reference_embeddings",
            "reference_prototypes",
            "output_dir",
            "bioclip_runtime_python",
            "bioclip_hf_cache_dir",
            "yoloe_runtime_python",
        ):
            value = Path(getattr(self, field)).expanduser()
            if "://" in str(value):
                raise ValueError(f"{field} must be a local path")
            object.__setattr__(self, field, value)
        for field in (
            "geography_sha256",
            "assignments_sha256",
            "query_hits_sha256",
            "regional_competitors_sha256",
            "readiness_sha256",
            "reference_embeddings_sha256",
            "reference_prototypes_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if self.storage_backend != "local" or self.s3_permitted:
            raise ValueError("staged Flickr prototype requires local-only storage")
        if not self.target_accepted_taxon_key.strip():
            raise ValueError("target_accepted_taxon_key must be non-empty")
        if not self.target_scientific_name.strip():
            raise ValueError("target_scientific_name must be non-empty")
        limits = tuple(int(value) for value in self.stage_limits)
        if not limits or any(value <= 0 for value in limits):
            raise ValueError("stage_limits must contain positive integers")
        if tuple(sorted(set(limits))) != limits:
            raise ValueError("stage_limits must be unique and strictly increasing")
        object.__setattr__(self, "stage_limits", limits)
        if self.device not in {"mps", "cpu"}:
            raise ValueError("staged Flickr prototype device must be mps or cpu")
        for field in (
            "download_workers",
            "bioclip_batch_size",
            "yoloe_batch_size",
            "preprocess_workers",
            "max_record_attempts",
            "request_retries",
            "max_image_bytes",
        ):
            value = int(getattr(self, field))
            if value <= 0 and field != "request_retries":
                raise ValueError(f"{field} must be positive")
            if value < 0:
                raise ValueError(f"{field} must be non-negative")
        if not 0.0 <= self.max_failure_rate < 1.0:
            raise ValueError("max_failure_rate must be in [0, 1)")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")

    @classmethod
    def read_json(cls, path: str | Path) -> PrototypeStagedFlickrConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("staged Flickr config must be an object")
        values = dict(payload)
        if values.pop("schema_version", None) != STAGED_FLICKR_VERSION:
            raise ValueError("unsupported staged Flickr config schema")
        values["stage_limits"] = tuple(values["stage_limits"])
        return cls(**values)

    @property
    def fingerprint(self) -> str:
        payload = {
            item.name: (
                str(value)
                if isinstance(value := getattr(self, item.name), Path)
                else value
            )
            for item in fields(self)
        }
        return canonical_semantic_fingerprint(payload)


@dataclass(frozen=True, slots=True)
class StagedFlickrImage:
    flickr_photo_id: str
    image_url: str
    path: Path
    source_image_sha256: str
    decoded_image_sha256: str
    width: int
    height: int
    decoded: DecodedImage


@dataclass(frozen=True, slots=True)
class PrototypeStagedFlickrResult:
    report: dict[str, Any]
    results_path: Path
    candidates_path: Path
    failures_path: Path | None
    report_path: Path
    summary_path: Path
    state_path: Path


class FlickrImageFetcher(Protocol):
    def fetch(self, photo_id: str, cache_dir: Path) -> StagedFlickrImage: ...


class ImageEmbeddingScorer(Protocol):
    last_image_content_hashes: list[str] | None
    worker_process_starts: int
    model_load_count: int
    model_cache_hit_count: int
    model_refresh_count: int
    device: str | None
    gpu_name: str | None
    effective_image_resize_mode: str | None
    model_weights_sha256: str | None
    open_clip_config_sha256: str | None
    preprocessing_version: str | None
    preprocessing_fingerprint: str | None

    @property
    def cache_metrics(self) -> Mapping[str, object]: ...

    @property
    def memory_metrics(self) -> Mapping[str, object]: ...

    def ensure_model_attestation(self) -> None: ...

    def pin_reference_model_identity(self, **kwargs: str) -> None: ...

    def embed_text_labels(self, labels: Sequence[str]) -> list[list[float]]: ...

    def embed_image_paths(self, image_paths: Sequence[Path]) -> list[list[float]]: ...

    def close(self) -> None: ...


class RouteDetector(Protocol):
    worker_process_starts: int
    worker_request_count: int
    model_id: str
    model_version: str
    checkpoint: str
    prompt_set_fingerprint: str

    def detect_batch(
        self, images: Sequence[DecodedImage]
    ) -> list[list[DetectionCandidate]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _CandidateClass:
    class_kind: str
    class_id: str
    accepted_taxon_key: str | None
    display_name: str
    text_prompt: str
    target_candidate: bool
    candidate_reason: str


class FlickrRestImageFetcher:
    """Fetch one Flickr image without retaining a raw API response."""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float,
        retries: int,
        max_image_bytes: int,
    ) -> None:
        if not str(api_key).strip():
            raise ValueError("FLICKR_API_KEY is required")
        self._api_key = api_key
        self._timeout = float(timeout_seconds)
        self._retries = int(retries)
        self._max_image_bytes = int(max_image_bytes)

    def fetch(self, photo_id: str, cache_dir: Path) -> StagedFlickrImage:
        metadata = self._photo_metadata(photo_id)
        server = _required_text(metadata.get("server"), field="Flickr server")
        secret = _required_text(metadata.get("secret"), field="Flickr secret")
        if str(metadata.get("media") or "photo") != "photo":
            raise ValueError("Flickr record is not a still photo")
        urls = (
            f"https://live.staticflickr.com/{server}/{photo_id}_{secret}_z.jpg",
            f"https://live.staticflickr.com/{server}/{photo_id}_{secret}.jpg",
        )
        last_error: Exception | None = None
        for image_url in urls:
            try:
                return self._download(photo_id, image_url, cache_dir)
            except (OSError, ValueError, HTTPError, URLError) as exc:
                last_error = exc
        raise OSError(f"Flickr image variants failed: {last_error}")

    def _photo_metadata(self, photo_id: str) -> Mapping[str, object]:
        params = {
            "method": "flickr.photos.getInfo",
            "api_key": self._api_key,
            "photo_id": photo_id,
            "format": "json",
            "nojsoncallback": "1",
        }
        request = Request(
            "https://www.flickr.com/services/rest/?" + urlencode(params),
            headers={"User-Agent": "BioMiner/0.1 staged-prototype"},
        )
        payload = self._request_json(request)
        if payload.get("stat") != "ok" or not isinstance(payload.get("photo"), Mapping):
            message = str(payload.get("message") or "Flickr getInfo failed")
            raise OSError(message)
        photo = dict(payload["photo"])
        if str(photo.get("id") or "") != photo_id:
            raise ValueError("Flickr returned a different photo ID")
        return photo

    def _request_json(self, request: Request) -> Mapping[str, object]:
        for attempt in range(self._retries + 1):
            try:
                with urlopen(request, timeout=self._timeout) as response:
                    payload = json.load(response)
                if not isinstance(payload, Mapping):
                    raise ValueError("Flickr response must be an object")
                return payload
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt >= self._retries:
                    raise OSError(
                        f"Flickr metadata request failed: {type(exc).__name__}"
                    ) from exc
                sleep(min(4.0, 0.5 * (2**attempt)))
        raise AssertionError("unreachable")

    def _download(
        self, photo_id: str, image_url: str, cache_dir: Path
    ) -> StagedFlickrImage:
        request = Request(
            image_url,
            headers={"User-Agent": "BioMiner/0.1 staged-prototype"},
        )
        body: bytes | None = None
        for attempt in range(self._retries + 1):
            try:
                with urlopen(request, timeout=self._timeout) as response:
                    content_type = str(response.headers.get_content_type())
                    if not content_type.startswith("image/"):
                        raise ValueError("Flickr response content type is not image/*")
                    body = response.read(self._max_image_bytes + 1)
                if len(body) > self._max_image_bytes:
                    raise ValueError("Flickr image exceeds configured byte limit")
                if not body:
                    raise ValueError("Flickr image is empty")
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                if attempt >= self._retries:
                    raise OSError(
                        f"Flickr image request failed: {type(exc).__name__}"
                    ) from exc
                sleep(min(4.0, 0.5 * (2**attempt)))
        assert body is not None
        source_sha = "sha256:" + hashlib.sha256(body).hexdigest()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{source_sha.removeprefix('sha256:')}.image"
        if not path.exists():
            temporary = path.with_suffix(f".tmp-{os.getpid()}-{photo_id}")
            temporary.write_bytes(body)
            os.replace(temporary, path)
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                decoded_sha = decoded_rgb_image_content_hash(rgb)
                decoded = DecodedImage(
                    width=width,
                    height=height,
                    mode="RGB",
                    data=rgb.tobytes(),
                    source_uri=image_url,
                )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return StagedFlickrImage(
            flickr_photo_id=photo_id,
            image_url=image_url,
            path=path,
            source_image_sha256=source_sha,
            decoded_image_sha256=decoded_sha,
            width=width,
            height=height,
            decoded=decoded,
        )


class _CheckpointStore:
    def __init__(
        self,
        path: Path,
        *,
        config_fingerprint: str,
        source_fingerprint: str,
        resume: bool,
        retry_failed: bool,
        max_record_attempts: int,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not resume:
            raise FileExistsError(f"checkpoint already exists: {path}")
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_identity (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                config_fingerprint TEXT NOT NULL,
                source_fingerprint TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS records (
                order_index INTEGER PRIMARY KEY,
                flickr_photo_id TEXT NOT NULL UNIQUE,
                input_json TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                result_json TEXT,
                candidates_json TEXT,
                failure_json TEXT,
                updated_at TEXT NOT NULL
            );
            """
        )
        identity = self._conn.execute(
            "SELECT config_fingerprint, source_fingerprint, created_at FROM run_identity WHERE singleton=1"
        ).fetchone()
        if identity is None:
            created_at = _utc_now()
            self._conn.execute(
                "INSERT INTO run_identity VALUES (1, ?, ?, ?)",
                (config_fingerprint, source_fingerprint, created_at),
            )
            self.created_at = created_at
        elif (
            identity["config_fingerprint"] != config_fingerprint
            or identity["source_fingerprint"] != source_fingerprint
        ):
            raise ValueError("staged Flickr checkpoint identity mismatch")
        else:
            self.created_at = str(identity["created_at"])
        self._conn.execute(
            "UPDATE records SET status='pending' WHERE status='in_progress'"
        )
        if retry_failed:
            self._conn.execute(
                """
                UPDATE records SET status='pending'
                WHERE status='failed' AND attempts < ?
                """,
                (max_record_attempts,),
            )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def seed(self, rows: Sequence[Mapping[str, object]]) -> None:
        now = _utc_now()
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO records(
                order_index, flickr_photo_id, input_json, status, updated_at
            ) VALUES (?, ?, ?, 'pending', ?)
            """,
            (
                (
                    index,
                    str(row["flickr_photo_id"]),
                    json.dumps(row, sort_keys=True, separators=(",", ":")),
                    now,
                )
                for index, row in enumerate(rows, start=1)
            ),
        )
        count = int(self._conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])
        if count != len(rows):
            raise ValueError("checkpoint record set differs from staged workload")
        self._conn.commit()

    def claim(self, *, stage_limit: int, batch_size: int) -> list[dict[str, object]]:
        rows = self._conn.execute(
            """
            SELECT order_index, flickr_photo_id, input_json, attempts
            FROM records
            WHERE order_index <= ? AND status='pending'
            ORDER BY order_index
            LIMIT ?
            """,
            (stage_limit, batch_size),
        ).fetchall()
        if not rows:
            return []
        now = _utc_now()
        self._conn.executemany(
            "UPDATE records SET status='in_progress', attempts=attempts+1, updated_at=? WHERE order_index=?",
            ((now, int(row["order_index"])) for row in rows),
        )
        self._conn.commit()
        output = []
        for row in rows:
            value = json.loads(str(row["input_json"]))
            value["order_index"] = int(row["order_index"])
            value["attempt"] = int(row["attempts"]) + 1
            output.append(value)
        return output

    def complete(
        self,
        *,
        order_index: int,
        result: Mapping[str, object],
        candidates: Sequence[Mapping[str, object]],
    ) -> None:
        self._conn.execute(
            """
            UPDATE records SET status='complete', result_json=?, candidates_json=?,
                failure_json=NULL, updated_at=? WHERE order_index=?
            """,
            (
                json.dumps(result, sort_keys=True, separators=(",", ":")),
                json.dumps(candidates, sort_keys=True, separators=(",", ":")),
                _utc_now(),
                order_index,
            ),
        )
        self._conn.commit()

    def fail(self, *, order_index: int, failure: Mapping[str, object]) -> None:
        self._conn.execute(
            """
            UPDATE records SET status='failed', failure_json=?, updated_at=?
            WHERE order_index=?
            """,
            (
                json.dumps(failure, sort_keys=True, separators=(",", ":")),
                _utc_now(),
                order_index,
            ),
        )
        self._conn.commit()

    def counts(self, stage_limit: int) -> dict[str, int]:
        rows = self._conn.execute(
            """
            SELECT status, COUNT(*) AS count FROM records
            WHERE order_index <= ? GROUP BY status
            """,
            (stage_limit,),
        ).fetchall()
        output = {"pending": 0, "in_progress": 0, "complete": 0, "failed": 0}
        output.update({str(row["status"]): int(row["count"]) for row in rows})
        return output

    def materialized_rows(
        self,
    ) -> tuple[
        list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]
    ]:
        results: list[dict[str, object]] = []
        candidates: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        rows = self._conn.execute(
            """
            SELECT order_index, attempts, status, result_json, candidates_json, failure_json
            FROM records WHERE status IN ('complete', 'failed') ORDER BY order_index
            """
        ).fetchall()
        for row in rows:
            if row["status"] == "complete":
                result = json.loads(str(row["result_json"]))
                result["attempt"] = int(row["attempts"])
                results.append(result)
                candidates.extend(json.loads(str(row["candidates_json"])))
            else:
                failure = json.loads(str(row["failure_json"]))
                failure["attempt"] = int(row["attempts"])
                failures.append(failure)
        return results, candidates, failures


def run_prototype_staged_flickr(
    config: PrototypeStagedFlickrConfig,
    *,
    flickr_fetcher: FlickrImageFetcher | None = None,
    scorer: ImageEmbeddingScorer | None = None,
    detector: RouteDetector | None = None,
) -> PrototypeStagedFlickrResult:
    """Run cumulative P1/P2/P3 gates using local state and temporary images."""

    started_at = _utc_now()
    started = perf_counter()
    _verify_inputs(config)
    readiness = json.loads(config.readiness.read_text(encoding="utf-8"))
    if readiness.get("bank_status") != "prototype_only" or not readiness.get(
        "classification_authorised"
    ):
        raise ValueError("prototype readiness does not authorise classification")
    workload = _load_workload(config)
    if config.stage_limits[-1] > len(workload):
        raise ValueError(
            f"final stage requires {config.stage_limits[-1]} records; workload has {len(workload)}"
        )
    workload = workload[: config.stage_limits[-1]]
    reference_embeddings = pl.read_parquet(config.reference_embeddings)
    reference_prototypes = pl.read_parquet(config.reference_prototypes)
    _validate_reference_inputs(
        reference_embeddings,
        reference_prototypes,
        target_accepted_taxon_key=config.target_accepted_taxon_key,
    )
    classes = _candidate_classes(config, reference_embeddings)
    source_fingerprint = canonical_semantic_fingerprint(
        [row["source_record_hash"] for row in workload]
    )
    output_dir = config.output_dir
    cache_dir = output_dir / "cache" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / STAGED_FLICKR_STATE_FILE
    checkpoint = _CheckpointStore(
        state_path,
        config_fingerprint=config.fingerprint,
        source_fingerprint=source_fingerprint,
        resume=config.resume,
        retry_failed=config.retry_failed,
        max_record_attempts=config.max_record_attempts,
    )
    checkpoint.seed(workload)

    results_path = output_dir / STAGED_FLICKR_RESULTS_FILE
    candidates_path = output_dir / STAGED_FLICKR_CANDIDATES_FILE
    failures_path = output_dir / STAGED_FLICKR_FAILURES_FILE
    report_path = output_dir / STAGED_FLICKR_REPORT_FILE
    summary_path = output_dir / STAGED_FLICKR_SUMMARY_FILE
    final_counts = checkpoint.counts(config.stage_limits[-1])
    classified_before_run = int(final_counts["complete"])
    if (
        final_counts["complete"] + final_counts["failed"] == config.stage_limits[-1]
        and final_counts["pending"] == 0
        and final_counts["in_progress"] == 0
        and report_path.is_file()
    ):
        checkpoint.close()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("configuration_fingerprint") != config.fingerprint
            or report.get("source_fingerprint") != source_fingerprint
            or report.get("status") != "passed"
        ):
            raise ValueError("completed staged Flickr report identity mismatch")
        for path in (results_path, candidates_path, failures_path, summary_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"completed staged Flickr artifact is missing: {path}"
                )
        return PrototypeStagedFlickrResult(
            report=report,
            results_path=results_path,
            candidates_path=candidates_path,
            failures_path=failures_path,
            report_path=report_path,
            summary_path=summary_path,
            state_path=state_path,
        )

    if flickr_fetcher is None:
        api_key = os.environ.get("FLICKR_API_KEY")
        if not api_key:
            raise ValueError("FLICKR_API_KEY is required for staged Flickr inference")
        flickr_fetcher = FlickrRestImageFetcher(
            api_key=api_key,
            timeout_seconds=config.request_timeout_seconds,
            retries=config.request_retries,
            max_image_bytes=config.max_image_bytes,
        )
    owns_scorer = scorer is None
    if scorer is None:
        scorer = _bioclip_scorer(config)
    owns_detector = detector is None
    if detector is None:
        detector = _yoloe_detector(config)

    stages: list[dict[str, object]] = []
    previous_report = _read_previous_report(report_path)
    text_embeddings: dict[str, list[float]] = {}
    model_started = perf_counter()
    try:
        scorer.ensure_model_attestation()
        _pin_reference_identity(scorer, reference_embeddings)
        embedded_text = scorer.embed_text_labels([item.text_prompt for item in classes])
        if len(embedded_text) != len(classes):
            raise RuntimeError("BioCLIP returned an incomplete text embedding set")
        text_embeddings = {
            item.class_id: _unit_vector(values)
            for item, values in zip(classes, embedded_text, strict=True)
        }
        for stage_number, stage_limit in enumerate(config.stage_limits, start=1):
            stage_started = perf_counter()
            initial_counts = checkpoint.counts(stage_limit)
            classified_before_stage = int(initial_counts["complete"])
            if (
                initial_counts["pending"] == 0
                and initial_counts["in_progress"] == 0
                and initial_counts["complete"] + initial_counts["failed"] == stage_limit
            ):
                previous_stage = _previous_stage(
                    previous_report, stage_number=stage_number, stage_limit=stage_limit
                )
                prior = {
                    key: value
                    for key, value in previous_stage.items()
                    if key
                    not in {
                        "resume_validation_checks",
                        "resumed_records_classified",
                        "resumed_retry_without_new_classification",
                        "resumed_without_stage_work",
                    }
                }
                stages.append({**prior, "resumed_without_stage_work": True})
                continue
            while True:
                batch = checkpoint.claim(
                    stage_limit=stage_limit,
                    batch_size=max(config.bioclip_batch_size, config.yoloe_batch_size),
                )
                if not batch:
                    break
                _process_batch(
                    batch=batch,
                    checkpoint=checkpoint,
                    fetcher=flickr_fetcher,
                    scorer=scorer,
                    detector=detector,
                    classes=classes,
                    text_embeddings=text_embeddings,
                    reference_embeddings=reference_embeddings,
                    reference_prototypes=reference_prototypes,
                    cache_dir=cache_dir,
                    config=config,
                )
                _write_progress(
                    output_dir / STAGED_FLICKR_PROGRESS_FILE,
                    stage_number=stage_number,
                    stage_limit=stage_limit,
                    counts=checkpoint.counts(stage_limit),
                    elapsed_seconds=perf_counter() - started,
                )
            final_stage_counts = checkpoint.counts(stage_limit)
            if detector.worker_process_starts == 0:
                previous_stage = _previous_stage(
                    previous_report, stage_number=stage_number, stage_limit=stage_limit
                )
                failures = int(final_stage_counts["failed"])
                prior = {
                    key: value
                    for key, value in previous_stage.items()
                    if key
                    not in {
                        "resume_validation_checks",
                        "resumed_records_classified",
                        "resumed_retry_without_new_classification",
                        "resumed_without_stage_work",
                    }
                }
                stages.append(
                    {
                        **prior,
                        "classified": int(final_stage_counts["complete"]),
                        "failures": failures,
                        "failure_rate": round(failures / stage_limit, 8),
                        "resumed_retry_without_new_classification": True,
                    }
                )
                continue
            stage = _validate_stage(
                stage_number=stage_number,
                stage_limit=stage_limit,
                counts=final_stage_counts,
                rows=checkpoint.materialized_rows()[0],
                expected_candidate_count=sum(
                    item.class_kind == "species" for item in classes
                ),
                scorer=scorer,
                detector=detector,
                max_failure_rate=config.max_failure_rate,
                elapsed_seconds=perf_counter() - stage_started,
            )
            if previous_report is not None and classified_before_stage > 0:
                previous_stage = _previous_stage(
                    previous_report, stage_number=stage_number, stage_limit=stage_limit
                )
                failures = int(final_stage_counts["failed"])
                prior = {
                    key: value
                    for key, value in previous_stage.items()
                    if key
                    not in {
                        "resume_validation_checks",
                        "resumed_records_classified",
                        "resumed_retry_without_new_classification",
                        "resumed_without_stage_work",
                    }
                }
                stages.append(
                    {
                        **prior,
                        "classified": int(final_stage_counts["complete"]),
                        "failures": failures,
                        "failure_rate": round(failures / stage_limit, 8),
                        "resume_validation_checks": stage["checks"],
                        "resumed_records_classified": int(
                            final_stage_counts["complete"] - classified_before_stage
                        ),
                    }
                )
            else:
                stages.append(stage)
    finally:
        if owns_scorer:
            scorer.close()
        if owns_detector:
            detector.close()
        _remove_cache(cache_dir)

    result_rows, candidate_rows, failure_rows = checkpoint.materialized_rows()
    run_started_at = checkpoint.created_at
    _write_outputs(
        results=result_rows,
        candidates=candidate_rows,
        failures=failure_rows,
        results_path=results_path,
        candidates_path=candidates_path,
        failures_path=failures_path,
    )
    checkpoint.close()
    model_seconds = perf_counter() - model_started
    reuse_previous_runtime = (
        previous_report is not None
        and len(result_rows) == classified_before_run
        and detector.worker_process_starts == 0
    )
    previous_memory = (
        previous_report.get("memory") if previous_report is not None else None
    )
    memory = (
        dict(previous_memory)
        if reuse_previous_runtime and isinstance(previous_memory, Mapping)
        else _memory_report(scorer, stages)
    )
    model_report = (
        dict(previous_report["model"])
        if reuse_previous_runtime
        and previous_report is not None
        and isinstance(previous_report.get("model"), Mapping)
        else _model_report(scorer, config, model_seconds)
    )
    detector_report = (
        dict(previous_report["detector"])
        if reuse_previous_runtime
        and previous_report is not None
        and isinstance(previous_report.get("detector"), Mapping)
        else _detector_report(detector)
    )
    elapsed_seconds = (
        float(previous_report["elapsed_seconds"])
        if reuse_previous_runtime
        and previous_report is not None
        and isinstance(previous_report.get("elapsed_seconds"), int | float)
        else sum(float(stage["elapsed_seconds"]) for stage in stages)
    )
    report: dict[str, Any] = {
        "schema_version": STAGED_FLICKR_REPORT_VERSION,
        "status": "passed",
        "started_at": run_started_at,
        "report_invocation_started_at": started_at,
        "ended_at": _utc_now(),
        "configuration_fingerprint": config.fingerprint,
        "source_fingerprint": source_fingerprint,
        "storage": {
            "backend": "local",
            "s3_permitted": False,
            "s3_accessed": False,
        },
        "prototype_only": True,
        "experimental_screening_evidence_only": True,
        "target": {
            "accepted_taxon_key": config.target_accepted_taxon_key,
            "scientific_name": config.target_scientific_name,
        },
        "counts": {
            "planned": len(workload),
            "classified": len(result_rows),
            "failures": len(failure_rows),
            "candidate_score_rows": len(candidate_rows),
        },
        "stages": stages,
        "candidate_union": {
            "species_count": sum(item.class_kind == "species" for item in classes),
            "known_negative_count": sum(
                item.class_kind == "known_negative" for item in classes
            ),
            "visual_domain_count": sum(
                item.class_kind == "visual_domain" for item in classes
            ),
            "target_always_scored": True,
            "higher_rank_pruning_applied": False,
        },
        "visual_input": {
            "kind": RAW_FULL_IMAGE,
            "complete_canvas_retained": True,
            "spatial_crop_applied": False,
        },
        "model": model_report,
        "detector": detector_report,
        "memory": memory,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "elapsed_seconds_scope": "sum_of_stage_gate_elapsed_seconds",
        "resume": {
            "reused_previous_runtime_evidence": reuse_previous_runtime,
            "classified_before_run": classified_before_run,
            "classified_after_run": len(result_rows),
        },
        "semantics": {
            "scores_are_probabilities": False,
            "scores_are_taxonomic_validation": False,
            "flickr_query_match_is_label": False,
            "human_review_required_to_report_accuracy": True,
            "score_semantics": SCORE_SEMANTICS,
        },
        "artifacts": {
            "results": _artifact(results_path),
            "candidates": _artifact(candidates_path),
            "failures": _artifact(failures_path) if failures_path.exists() else None,
            "state": _artifact(state_path),
        },
    }
    report["report_fingerprint"] = canonical_semantic_fingerprint(report)
    _atomic_write_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_write_text(summary_path, _summary(report))
    return PrototypeStagedFlickrResult(
        report=report,
        results_path=results_path,
        candidates_path=candidates_path,
        failures_path=failures_path if failures_path.exists() else None,
        report_path=report_path,
        summary_path=summary_path,
        state_path=state_path,
    )


def _process_batch(
    *,
    batch: Sequence[Mapping[str, object]],
    checkpoint: _CheckpointStore,
    fetcher: FlickrImageFetcher,
    scorer: ImageEmbeddingScorer,
    detector: RouteDetector,
    classes: Sequence[_CandidateClass],
    text_embeddings: Mapping[str, Sequence[float]],
    reference_embeddings: pl.DataFrame,
    reference_prototypes: pl.DataFrame,
    cache_dir: Path,
    config: PrototypeStagedFlickrConfig,
) -> None:
    staged: dict[int, StagedFlickrImage] = {}
    with ThreadPoolExecutor(max_workers=config.download_workers) as executor:
        future_rows = {
            executor.submit(fetcher.fetch, str(row["flickr_photo_id"]), cache_dir): row
            for row in batch
        }
        for future in as_completed(future_rows):
            row = future_rows[future]
            index = int(row["order_index"])
            try:
                staged[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - one source failure remains retryable.
                checkpoint.fail(
                    order_index=index,
                    failure=_failure(row, "download_or_decode", exc),
                )
    ordered_rows = [row for row in batch if int(row["order_index"]) in staged]
    if not ordered_rows:
        return
    images = [staged[int(row["order_index"])] for row in ordered_rows]
    try:
        detection_results: dict[str, list[DetectionCandidate] | Exception] = {}
        for start in range(0, len(images), config.yoloe_batch_size):
            detection_results.update(
                _detect_resilient(
                    detector, images[start : start + config.yoloe_batch_size]
                )
            )
        embedding_results: dict[str, list[float] | Exception] = {}
        for start in range(0, len(images), config.bioclip_batch_size):
            embedding_results.update(
                _embed_resilient(
                    scorer, images[start : start + config.bioclip_batch_size]
                )
            )
        for row, image in zip(ordered_rows, images, strict=True):
            index = int(row["order_index"])
            photo_id = str(row["flickr_photo_id"])
            detection = detection_results.get(photo_id)
            embedding = embedding_results.get(photo_id)
            if isinstance(detection, Exception):
                checkpoint.fail(
                    order_index=index,
                    failure=_failure(row, "yoloe", detection),
                )
                continue
            if isinstance(embedding, Exception):
                checkpoint.fail(
                    order_index=index,
                    failure=_failure(row, "bioclip", embedding),
                )
                continue
            try:
                result, candidates = _score_record(
                    row=row,
                    image=image,
                    detections=detection,
                    image_embedding=embedding,
                    classes=classes,
                    text_embeddings=text_embeddings,
                    reference_embeddings=reference_embeddings,
                    reference_prototypes=reference_prototypes,
                    target_accepted_taxon_key=config.target_accepted_taxon_key,
                    target_scientific_name=config.target_scientific_name,
                )
                checkpoint.complete(
                    order_index=index,
                    result=result,
                    candidates=candidates,
                )
            except Exception as exc:  # noqa: BLE001 - quarantine scoring defect per record.
                checkpoint.fail(
                    order_index=index,
                    failure=_failure(row, "scoring", exc),
                )
    finally:
        for image in images:
            image.path.unlink(missing_ok=True)


def _detect_resilient(
    detector: RouteDetector,
    images: Sequence[StagedFlickrImage],
) -> dict[str, list[DetectionCandidate] | Exception]:
    output: dict[str, list[DetectionCandidate] | Exception] = {}

    def visit(values: Sequence[StagedFlickrImage]) -> None:
        if not values:
            return
        try:
            detections = detector.detect_batch([item.decoded for item in values])
            if len(detections) != len(values):
                raise RuntimeError("YOLOE returned the wrong batch length")
            for item, rows in zip(values, detections, strict=True):
                output[item.flickr_photo_id] = rows
        except Exception as exc:  # noqa: BLE001 - split a bad model batch.
            if len(values) == 1:
                output[values[0].flickr_photo_id] = exc
                return
            midpoint = len(values) // 2
            visit(values[:midpoint])
            visit(values[midpoint:])

    visit(images)
    return output


def _embed_resilient(
    scorer: ImageEmbeddingScorer,
    images: Sequence[StagedFlickrImage],
) -> dict[str, list[float] | Exception]:
    output: dict[str, list[float] | Exception] = {}

    def visit(values: Sequence[StagedFlickrImage]) -> None:
        if not values:
            return
        try:
            embeddings = scorer.embed_image_paths([item.path for item in values])
            if len(embeddings) != len(values):
                raise RuntimeError("BioCLIP returned the wrong batch length")
            expected_hashes = [item.decoded_image_sha256 for item in values]
            if scorer.last_image_content_hashes != expected_hashes:
                raise RuntimeError(
                    "BioCLIP decoded content hashes differ from controller"
                )
            for item, values_ in zip(values, embeddings, strict=True):
                output[item.flickr_photo_id] = _unit_vector(values_)
        except Exception as exc:  # noqa: BLE001 - split a bad model batch.
            if len(values) == 1:
                output[values[0].flickr_photo_id] = exc
                return
            midpoint = len(values) // 2
            visit(values[:midpoint])
            visit(values[midpoint:])

    visit(images)
    return output


def _score_record(
    *,
    row: Mapping[str, object],
    image: StagedFlickrImage,
    detections: Sequence[DetectionCandidate],
    image_embedding: Sequence[float],
    classes: Sequence[_CandidateClass],
    text_embeddings: Mapping[str, Sequence[float]],
    reference_embeddings: pl.DataFrame,
    reference_prototypes: pl.DataFrame,
    target_accepted_taxon_key: str,
    target_scientific_name: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    embedding = _unit_vector(image_embedding)
    route_fields = _route_fields(detections)
    bioclip_route = route_fields["bioclip_route"]
    route_support = (
        reference_embeddings.filter(
            (pl.col("dataset_split") == "support_train")
            & (pl.col("route") == bioclip_route)
        )
        if bioclip_route is not None
        else reference_embeddings.head(0)
    )
    route_prototypes = (
        reference_prototypes.filter(
            (pl.col("scope_type") == "global") & (pl.col("route") == bioclip_route)
        )
        if bioclip_route is not None
        else reference_prototypes.head(0)
    )

    prototype_by_key = {
        str(item["accepted_taxon_key"]): item
        for item in route_prototypes.iter_rows(named=True)
    }
    candidate_rows: list[dict[str, object]] = []
    for item in classes:
        text_similarity = _dot(embedding, text_embeddings[item.class_id])
        prototype = (
            prototype_by_key.get(str(item.accepted_taxon_key))
            if item.accepted_taxon_key is not None
            else None
        )
        reference_similarity = (
            _dot(embedding, prototype["embedding"]) if prototype is not None else None
        )
        decision_score = (
            (0.75 * reference_similarity) + (0.25 * text_similarity)
            if reference_similarity is not None
            else text_similarity
        )
        candidate_rows.append(
            {
                "schema_version": STAGED_FLICKR_CANDIDATE_VERSION,
                "order_index": int(row["order_index"]),
                "flickr_photo_id": str(row["flickr_photo_id"]),
                "class_kind": item.class_kind,
                "class_id": item.class_id,
                "accepted_taxon_key": item.accepted_taxon_key,
                "display_name": item.display_name,
                "text_prompt": item.text_prompt,
                "candidate_reason": item.candidate_reason,
                "target_candidate": item.target_candidate,
                "text_similarity": text_similarity,
                "reference_prototype_similarity": reference_similarity,
                "prototype_score": decision_score,
                "rank": None,
                "score_semantics": SCORE_SEMANTICS,
                "experimental_screening_evidence": True,
            }
        )
    species = [item for item in candidate_rows if item["class_kind"] == "species"]
    ranked_species = sorted(
        species,
        key=lambda item: (-float(item["prototype_score"]), str(item["class_id"])),
    )
    for rank, candidate in enumerate(ranked_species, start=1):
        candidate["rank"] = rank
    target = next(item for item in species if item["target_candidate"])
    competitors = [item for item in ranked_species if not item["target_candidate"]]
    best_text_competitor = max(
        competitors,
        key=lambda item: (float(item["text_similarity"]), str(item["class_id"])),
    )
    best_scored_competitor = competitors[0]

    reference_metrics = _reference_metrics(
        embedding=embedding,
        route_support=route_support,
        route_prototypes=route_prototypes,
        geo_cluster_id=str(row["geo_cluster_id"]),
        target_accepted_taxon_key=target_accepted_taxon_key,
    )
    uncalibrated_margin = float(target["prototype_score"]) - float(
        best_scored_competitor["prototype_score"]
    )
    abstention_reason = _abstention_reason(
        route_fields=route_fields,
        reference_metrics=reference_metrics,
        margin=uncalibrated_margin,
    )
    detector_winner = (
        max(detections, key=lambda item: (item.score, item.label))
        if detections
        else None
    )
    result: dict[str, object] = {
        "schema_version": STAGED_FLICKR_RESULT_VERSION,
        "order_index": int(row["order_index"]),
        "flickr_photo_id": str(row["flickr_photo_id"]),
        "source": "flickr",
        "source_record_hash": str(row["source_record_hash"]),
        "geo_cluster_id": str(row["geo_cluster_id"]),
        "coordinate_quality": str(row["coordinate_quality"]),
        "query_terms": list(row["query_terms"]),
        "query_tiers": list(row["query_tiers"]),
        "flickr_query_match_is_label": False,
        "image_url": image.image_url,
        "source_image_sha256": image.source_image_sha256,
        "decoded_image_sha256": image.decoded_image_sha256,
        "decoded_width": image.width,
        "decoded_height": image.height,
        "visual_input": RAW_FULL_IMAGE,
        "complete_canvas_retained": True,
        "spatial_crop_applied": False,
        "detection_count": len(detections),
        "detector_label": detector_winner.label if detector_winner else None,
        "detector_prompt": detector_winner.detector_prompt if detector_winner else None,
        "detector_score": detector_winner.score if detector_winner else None,
        **route_fields,
        "reference_route_used": bioclip_route if not route_support.is_empty() else None,
        "target_accepted_taxon_key": target_accepted_taxon_key,
        "target_scientific_name": target_scientific_name,
        "target_scored": True,
        "regional_candidate_count": len(species),
        "regional_scored_count": len(species),
        "higher_rank_pruning_applied": False,
        "hierarchy_rankings_diagnostic_only": True,
        "target_text_similarity": float(target["text_similarity"]),
        "target_text_rank": _rank_by_field(species, target, "text_similarity"),
        "best_text_competitor_key": best_text_competitor["accepted_taxon_key"],
        "best_text_competitor_name": best_text_competitor["display_name"],
        "best_text_competitor_similarity": float(
            best_text_competitor["text_similarity"]
        ),
        "target_text_competitor_margin": float(target["text_similarity"])
        - float(best_text_competitor["text_similarity"]),
        **reference_metrics,
        "best_scored_competitor_key": best_scored_competitor["accepted_taxon_key"],
        "best_scored_competitor_name": best_scored_competitor["display_name"],
        "best_scored_competitor_score": float(
            best_scored_competitor["prototype_score"]
        ),
        "prototype_score": float(target["prototype_score"]),
        "uncalibrated_margin": uncalibrated_margin,
        "abstain": abstention_reason is not None,
        "abstention_reason": abstention_reason,
        "calibrated_probability": None,
        "score_semantics": SCORE_SEMANTICS,
        "prototype_status": "prototype_only",
        "experimental_screening_evidence": True,
    }
    result["result_fingerprint"] = canonical_semantic_fingerprint(result)
    return result, candidate_rows


def _reference_metrics(
    *,
    embedding: Sequence[float],
    route_support: pl.DataFrame,
    route_prototypes: pl.DataFrame,
    geo_cluster_id: str,
    target_accepted_taxon_key: str,
) -> dict[str, object]:
    if route_support.is_empty():
        return {
            "target_global_centroid_similarity": None,
            "target_regional_centroid_similarity": None,
            "nearest_target_reference_similarity": None,
            "target_top3_reference_similarity": None,
            "target_top5_reference_similarity": None,
            "nearest_target_reference_media_id": None,
            "best_competitor_reference_similarity": None,
            "best_competitor_reference_key": None,
            "best_competitor_reference_name": None,
            "target_competitor_reference_margin": None,
            "nearest_references_json": "[]",
        }
    scored = []
    for item in route_support.iter_rows(named=True):
        scored.append(
            {
                "reference_media_id": str(item["reference_media_id"]),
                "accepted_taxon_key": str(item["accepted_taxon_key"]),
                "scientific_name": str(item["scientific_name"]),
                "reference_group": str(item["reference_group"]),
                "similarity": _dot(embedding, item["embedding"]),
            }
        )
    scored.sort(key=lambda item: (-item["similarity"], item["reference_media_id"]))
    target_rows = [
        item
        for item in scored
        if item["accepted_taxon_key"] == target_accepted_taxon_key
    ]
    competitor_rows = [
        item
        for item in scored
        if item["accepted_taxon_key"] != target_accepted_taxon_key
    ]
    nearest_target = target_rows[0] if target_rows else None
    best_competitor = competitor_rows[0] if competitor_rows else None
    target_global = route_prototypes.filter(
        (pl.col("accepted_taxon_key") == target_accepted_taxon_key)
        & (pl.col("scope_type") == "global")
    )
    target_regional = route_prototypes.filter(
        (pl.col("accepted_taxon_key") == target_accepted_taxon_key)
        & (pl.col("scope_type") == "regional")
        & (pl.col("geo_cluster_id") == geo_cluster_id)
    )
    target_global_similarity = (
        _dot(embedding, target_global["embedding"][0])
        if target_global.height == 1
        else None
    )
    target_regional_similarity = (
        _dot(embedding, target_regional["embedding"][0])
        if target_regional.height == 1
        else None
    )
    nearest_similarity = nearest_target["similarity"] if nearest_target else None
    competitor_similarity = best_competitor["similarity"] if best_competitor else None
    return {
        "target_global_centroid_similarity": target_global_similarity,
        "target_regional_centroid_similarity": target_regional_similarity,
        "nearest_target_reference_similarity": nearest_similarity,
        "target_top3_reference_similarity": _mean_top_k(target_rows, 3),
        "target_top5_reference_similarity": _mean_top_k(target_rows, 5),
        "nearest_target_reference_media_id": (
            nearest_target["reference_media_id"] if nearest_target else None
        ),
        "best_competitor_reference_similarity": competitor_similarity,
        "best_competitor_reference_key": (
            best_competitor["accepted_taxon_key"] if best_competitor else None
        ),
        "best_competitor_reference_name": (
            best_competitor["scientific_name"] if best_competitor else None
        ),
        "target_competitor_reference_margin": (
            nearest_similarity - competitor_similarity
            if nearest_similarity is not None and competitor_similarity is not None
            else None
        ),
        "nearest_references_json": json.dumps(
            scored[:5], sort_keys=True, separators=(",", ":")
        ),
    }


def _route_fields(
    detections: Sequence[DetectionCandidate],
) -> dict[str, object]:
    if detections:
        winner = max(detections, key=lambda item: (item.score, item.label))
        fields = {
            "detection_status": "detected",
            "detector_label": winner.label,
            "detector_score": winner.score,
            "detector_prompt": winner.detector_prompt,
        }
    else:
        fields = {"detection_status": "no_detection"}
    return route_detection(fields).as_row_fields()


def _abstention_reason(
    *,
    route_fields: Mapping[str, object],
    reference_metrics: Mapping[str, object],
    margin: float,
) -> str | None:
    if route_fields["routing_action"] != "score":
        return f"route:{route_fields['detection_route']}"
    if reference_metrics["nearest_target_reference_similarity"] is None:
        return "compatible_route_has_no_target_support"
    if margin < 0.02:
        return "uncalibrated_margin_below_0.02"
    return None


def _candidate_classes(
    config: PrototypeStagedFlickrConfig,
    reference_embeddings: pl.DataFrame,
) -> tuple[_CandidateClass, ...]:
    evidence = pl.read_parquet(config.regional_competitors).sort(
        ["candidate_rank", "candidate_scientific_name"]
    )
    species: dict[str, _CandidateClass] = {
        config.target_accepted_taxon_key: _CandidateClass(
            class_kind="species",
            class_id=config.target_accepted_taxon_key,
            accepted_taxon_key=config.target_accepted_taxon_key,
            display_name=config.target_scientific_name,
            text_prompt=f"a field photograph of {config.target_scientific_name}",
            target_candidate=True,
            candidate_reason="target_always_scored",
        )
    }
    for item in evidence.iter_rows(named=True):
        key = str(item["candidate_accepted_taxon_key"])
        name = str(item["candidate_scientific_name"])
        species.setdefault(
            key,
            _CandidateClass(
                class_kind="species",
                class_id=key,
                accepted_taxon_key=key,
                display_name=name,
                text_prompt=f"a field photograph of {name}",
                target_candidate=False,
                candidate_reason=str(item["candidate_reason"]),
            ),
        )
    support_taxa = (
        reference_embeddings.filter(pl.col("dataset_split") == "support_train")
        .select(["accepted_taxon_key", "scientific_name", "reference_group"])
        .unique()
        .sort(["scientific_name", "accepted_taxon_key"])
    )
    for item in support_taxa.iter_rows(named=True):
        key = str(item["accepted_taxon_key"])
        name = str(item["scientific_name"])
        species.setdefault(
            key,
            _CandidateClass(
                class_kind="species",
                class_id=key,
                accepted_taxon_key=key,
                display_name=name,
                text_prompt=f"a field photograph of {name}",
                target_candidate=False,
                candidate_reason=f"prototype_support:{item['reference_group']}",
            ),
        )
    ordered_species = [
        species[config.target_accepted_taxon_key],
        *sorted(
            (
                item
                for key, item in species.items()
                if key != config.target_accepted_taxon_key
            ),
            key=lambda item: (item.display_name, item.class_id),
        ),
    ]
    auxiliary = [
        _CandidateClass(
            class_kind=kind,
            class_id=class_id,
            accepted_taxon_key=None,
            display_name=class_id.split(":", 1)[1].replace("-", " "),
            text_prompt=prompt,
            target_candidate=False,
            candidate_reason="explicit_auxiliary_class",
        )
        for kind, class_id, prompt in _AUXILIARY_TEXT_CLASSES
    ]
    return tuple([*ordered_species, *auxiliary])


def _load_workload(
    config: PrototypeStagedFlickrConfig,
) -> list[dict[str, object]]:
    geography = pl.read_parquet(config.geography)
    assignments = pl.read_parquet(config.assignments).select(
        ["flickr_photo_id", "geo_cluster_id"]
    )
    hits = (
        pl.read_parquet(config.query_hits)
        .group_by("flickr_photo_id")
        .agg(
            pl.col("search_term").unique().sort().alias("query_terms"),
            pl.col("query_tier").unique().sort().alias("query_tiers"),
        )
    )
    joined = geography.join(
        assignments, on="flickr_photo_id", how="inner", validate="1:1"
    ).join(hits, on="flickr_photo_id", how="inner", validate="1:1")
    if joined.height != geography.height:
        raise ValueError("Flickr workload joins are incomplete")
    rows = joined.select(
        [
            "flickr_photo_id",
            "source_record_hash",
            "geo_cluster_id",
            "coordinate_quality",
            "query_terms",
            "query_tiers",
        ]
    ).to_dicts()
    rows.sort(
        key=lambda item: (
            hashlib.sha256(str(item["flickr_photo_id"]).encode()).hexdigest(),
            str(item["flickr_photo_id"]),
        )
    )
    return rows


def _validate_reference_inputs(
    embeddings: pl.DataFrame,
    prototypes: pl.DataFrame,
    *,
    target_accepted_taxon_key: str,
) -> None:
    embedding_columns = {
        "reference_media_id",
        "accepted_taxon_key",
        "scientific_name",
        "reference_group",
        "route",
        "dataset_split",
        "embedding",
        "model_id",
        "model_revision",
        "model_weights_sha256",
        "open_clip_version",
        "open_clip_config_sha256",
        "preprocessing_fingerprint",
    }
    prototype_columns = {
        "accepted_taxon_key",
        "route",
        "scope_type",
        "geo_cluster_id",
        "embedding",
    }
    if missing := embedding_columns.difference(embeddings.columns):
        raise ValueError(f"reference embeddings missing columns: {sorted(missing)}")
    if missing := prototype_columns.difference(prototypes.columns):
        raise ValueError(f"reference prototypes missing columns: {sorted(missing)}")
    support = embeddings.filter(pl.col("dataset_split") == "support_train")
    if (
        support.is_empty()
        or support.filter(
            (pl.col("accepted_taxon_key") == target_accepted_taxon_key)
            & (pl.col("route") == "adult_field")
        ).is_empty()
    ):
        raise ValueError("support_train lacks adult target references")
    for frame in (embeddings, prototypes):
        for values in frame["embedding"]:
            _unit_vector(values)


def _verify_inputs(config: PrototypeStagedFlickrConfig) -> None:
    for path, expected in (
        (config.geography, config.geography_sha256),
        (config.assignments, config.assignments_sha256),
        (config.query_hits, config.query_hits_sha256),
        (config.regional_competitors, config.regional_competitors_sha256),
        (config.readiness, config.readiness_sha256),
        (config.reference_embeddings, config.reference_embeddings_sha256),
        (config.reference_prototypes, config.reference_prototypes_sha256),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if _file_sha256(path) != expected:
            raise ValueError(f"input SHA-256 mismatch: {path}")


def _pin_reference_identity(
    scorer: ImageEmbeddingScorer,
    embeddings: pl.DataFrame,
) -> None:
    identity = embeddings.row(0, named=True)
    scorer.pin_reference_model_identity(
        model_weights_sha256=str(identity["model_weights_sha256"]),
        open_clip_version=str(identity["open_clip_version"]),
        open_clip_config_sha256=str(identity["open_clip_config_sha256"]),
        preprocessing_fingerprint=str(identity["preprocessing_fingerprint"]),
        image_resize_mode="longest",
    )


def _failure(
    row: Mapping[str, object], stage: str, error: Exception
) -> dict[str, object]:
    message = " ".join(str(error).split())[:1000]
    return {
        "schema_version": STAGED_FLICKR_FAILURE_VERSION,
        "order_index": int(row["order_index"]),
        "flickr_photo_id": str(row["flickr_photo_id"]),
        "failure_stage": stage,
        "retryable": True,
        "error_type": type(error).__name__,
        "error": message,
        "biological_negative": False,
        "occurred_at": _utc_now(),
    }


def _mean_top_k(rows: Sequence[Mapping[str, object]], k: int) -> float | None:
    values = [float(item["similarity"]) for item in rows[:k]]
    return sum(values) / len(values) if values else None


def _rank_by_field(
    rows: Sequence[Mapping[str, object]],
    target: Mapping[str, object],
    field: str,
) -> int:
    ranked = sorted(
        rows,
        key=lambda item: (-float(item[field]), str(item["class_id"])),
    )
    return next(index for index, item in enumerate(ranked, start=1) if item is target)


def _unit_vector(values: Sequence[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result or any(not isfinite(value) for value in result):
        raise ValueError("embedding must contain finite values")
    norm = sqrt(sum(value * value for value in result))
    if not isfinite(norm) or norm <= 0:
        raise ValueError("embedding must have a positive finite norm")
    normalized = [value / norm for value in result]
    if abs(sqrt(sum(value * value for value in normalized)) - 1.0) > 1e-6:
        raise ValueError("embedding normalization failed")
    return normalized


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("cannot compare embeddings with different dimensions")
    value = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    if not isfinite(value):
        raise ValueError("embedding similarity is not finite")
    return max(-1.0, min(1.0, value))


def _validate_stage(
    *,
    stage_number: int,
    stage_limit: int,
    counts: Mapping[str, int],
    rows: Sequence[Mapping[str, object]],
    expected_candidate_count: int,
    scorer: ImageEmbeddingScorer,
    detector: RouteDetector,
    max_failure_rate: float,
    elapsed_seconds: float,
) -> dict[str, object]:
    terminal = int(counts["complete"]) + int(counts["failed"])
    if terminal != stage_limit or counts["pending"] or counts["in_progress"]:
        raise RuntimeError(
            f"stage P{stage_number} is incomplete: limit={stage_limit}, counts={dict(counts)}"
        )
    failure_rate = int(counts["failed"]) / stage_limit
    if failure_rate > max_failure_rate:
        raise RuntimeError(
            f"stage P{stage_number} failure rate {failure_rate:.6f} exceeds {max_failure_rate:.6f}"
        )
    stage_rows = [item for item in rows if int(item["order_index"]) <= stage_limit]
    if not stage_rows:
        raise RuntimeError(f"stage P{stage_number} has no completed inference rows")
    if any(
        not item["target_scored"]
        or item["higher_rank_pruning_applied"]
        or item["spatial_crop_applied"]
        or not item["complete_canvas_retained"]
        or item["visual_input"] != RAW_FULL_IMAGE
        for item in stage_rows
    ):
        raise RuntimeError("target/no-pruning/full-frame stage invariant failed")
    if any(
        int(item["regional_candidate_count"]) != expected_candidate_count
        or int(item["regional_scored_count"]) != expected_candidate_count
        for item in stage_rows
    ):
        raise RuntimeError("complete candidate-union scoring invariant failed")
    if any(
        item["reference_route_used"] is not None
        and item["reference_route_used"] != item["bioclip_route"]
        for item in stage_rows
    ):
        raise RuntimeError("reference route separation invariant failed")
    if not any(item["nearest_target_reference_media_id"] for item in stage_rows):
        raise RuntimeError("stage did not exercise target reference retrieval")
    finite_fields = (
        "target_text_similarity",
        "best_text_competitor_similarity",
        "target_text_competitor_margin",
        "prototype_score",
        "uncalibrated_margin",
    )
    if any(
        not isfinite(float(item[field]))
        for item in stage_rows
        for field in finite_fields
    ):
        raise RuntimeError("stage output contains non-finite decision evidence")
    cache = dict(scorer.cache_metrics)
    if (
        int(cache.get("bioclip_worker_process_starts") or 0) != 1
        or int(cache.get("bioclip_model_loads") or 0) != 1
        or int(cache.get("bioclip_model_cache_hits") or 0) < 1
    ):
        raise RuntimeError("stage did not demonstrate persistent BioCLIP model reuse")
    memory = dict(scorer.memory_metrics)
    if scorer.device == "mps" and any(
        not isinstance(memory.get(field), int)
        for field in (
            "mps_current_allocated_memory",
            "mps_driver_allocated_memory",
            "mps_recommended_max_memory",
        )
    ):
        raise RuntimeError("stage did not instrument required MPS memory counters")
    if detector.worker_process_starts != 1 or detector.worker_request_count < 1:
        raise RuntimeError("stage did not demonstrate persistent YOLOE worker reuse")
    elapsed = max(float(elapsed_seconds), 1e-9)
    classified = int(counts["complete"])
    return {
        "stage_id": f"P{stage_number}",
        "cumulative_limit": stage_limit,
        "status": "passed",
        "classified": classified,
        "failures": int(counts["failed"]),
        "failure_rate": round(failure_rate, 8),
        "records_per_second": round(classified / elapsed, 6),
        "seconds_per_record": round(elapsed / max(classified, 1), 6),
        "elapsed_seconds": round(elapsed, 6),
        "rss_peak_memory": _maxrss_bytes(resource.RUSAGE_SELF),
        "checks": {
            "target_never_pruned": True,
            "complete_candidate_union_scored": True,
            "raw_full_frame_only": True,
            "spatial_crop_applied": False,
            "persistent_bioclip_worker": True,
            "persistent_yoloe_worker": True,
            "sqlite_checkpoint_complete": True,
            "finite_scores": True,
            "reference_retrieval_exercised": True,
            "competitor_margins_finite": True,
            "route_separation": True,
            "throughput_instrumented": True,
            "memory_instrumented": True,
            "failure_rate_within_limit": True,
        },
    }


def _memory_report(
    scorer: ImageEmbeddingScorer,
    stages: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "rss_peak_memory": max(
            int(stage.get("rss_peak_memory") or 0) for stage in stages
        ),
        "child_rss_peak_memory": "not_instrumented",
        **dict(scorer.memory_metrics),
    }


def _write_outputs(
    *,
    results: Sequence[Mapping[str, object]],
    candidates: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
    results_path: Path,
    candidates_path: Path,
    failures_path: Path,
) -> None:
    result_frame = pl.from_dicts(results, schema=_results_schema(), strict=False).sort(
        "order_index"
    )
    candidate_frame = pl.from_dicts(
        candidates, schema=_candidates_schema(), strict=False
    ).sort(["order_index", "class_kind", "rank", "class_id"], nulls_last=True)
    failure_frame = pl.from_dicts(
        failures, schema=_failures_schema(), strict=False
    ).sort("order_index")
    _validate_results(result_frame, candidate_frame)
    _atomic_write_parquet(result_frame, results_path)
    _atomic_write_parquet(candidate_frame, candidates_path)
    _atomic_write_parquet(failure_frame, failures_path)


def _validate_results(results: pl.DataFrame, candidates: pl.DataFrame) -> None:
    if results.is_empty():
        raise ValueError("staged Flickr classification produced no result rows")
    if results["flickr_photo_id"].n_unique() != results.height:
        raise ValueError("staged Flickr results contain duplicate photo IDs")
    if results.filter(
        pl.col("target_scored").not_()
        | pl.col("higher_rank_pruning_applied")
        | pl.col("spatial_crop_applied")
        | pl.col("complete_canvas_retained").not_()
    ).height:
        raise ValueError("staged Flickr results violate target-aware invariants")
    candidate_counts = candidates.group_by("flickr_photo_id").agg(
        pl.col("class_kind").eq("species").sum().alias("species_count"),
        pl.col("target_candidate").sum().alias("target_count"),
    )
    expected = results.select(["flickr_photo_id", "regional_scored_count"]).join(
        candidate_counts, on="flickr_photo_id", how="left", validate="1:1"
    )
    if expected.filter(
        (pl.col("species_count") != pl.col("regional_scored_count"))
        | (pl.col("target_count") != 1)
    ).height:
        raise ValueError("candidate score output is incomplete")


def _results_schema() -> dict[str, pl.DataType]:
    optional_float = pl.Float64
    return {
        "schema_version": pl.String,
        "order_index": pl.UInt32,
        "flickr_photo_id": pl.String,
        "source": pl.String,
        "source_record_hash": pl.String,
        "geo_cluster_id": pl.String,
        "coordinate_quality": pl.String,
        "query_terms": pl.List(pl.String),
        "query_tiers": pl.List(pl.String),
        "flickr_query_match_is_label": pl.Boolean,
        "image_url": pl.String,
        "source_image_sha256": pl.String,
        "decoded_image_sha256": pl.String,
        "decoded_width": pl.UInt32,
        "decoded_height": pl.UInt32,
        "visual_input": pl.String,
        "complete_canvas_retained": pl.Boolean,
        "spatial_crop_applied": pl.Boolean,
        "detection_count": pl.UInt32,
        "detector_label": pl.String,
        "detector_prompt": pl.String,
        "detector_score": optional_float,
        "detection_route": pl.String,
        "routing_action": pl.String,
        "bioclip_route": pl.String,
        "routing_priority": pl.String,
        "routing_reason": pl.String,
        "routing_policy_version": pl.String,
        "routing_policy_fingerprint": pl.String,
        "reference_route_used": pl.String,
        "target_accepted_taxon_key": pl.String,
        "target_scientific_name": pl.String,
        "target_scored": pl.Boolean,
        "regional_candidate_count": pl.UInt32,
        "regional_scored_count": pl.UInt32,
        "higher_rank_pruning_applied": pl.Boolean,
        "hierarchy_rankings_diagnostic_only": pl.Boolean,
        "target_text_similarity": pl.Float64,
        "target_text_rank": pl.UInt32,
        "best_text_competitor_key": pl.String,
        "best_text_competitor_name": pl.String,
        "best_text_competitor_similarity": pl.Float64,
        "target_text_competitor_margin": pl.Float64,
        "target_global_centroid_similarity": optional_float,
        "target_regional_centroid_similarity": optional_float,
        "nearest_target_reference_similarity": optional_float,
        "target_top3_reference_similarity": optional_float,
        "target_top5_reference_similarity": optional_float,
        "nearest_target_reference_media_id": pl.String,
        "best_competitor_reference_similarity": optional_float,
        "best_competitor_reference_key": pl.String,
        "best_competitor_reference_name": pl.String,
        "target_competitor_reference_margin": optional_float,
        "nearest_references_json": pl.String,
        "best_scored_competitor_key": pl.String,
        "best_scored_competitor_name": pl.String,
        "best_scored_competitor_score": pl.Float64,
        "prototype_score": pl.Float64,
        "uncalibrated_margin": pl.Float64,
        "abstain": pl.Boolean,
        "abstention_reason": pl.String,
        "calibrated_probability": optional_float,
        "score_semantics": pl.String,
        "prototype_status": pl.String,
        "experimental_screening_evidence": pl.Boolean,
        "result_fingerprint": pl.String,
        "attempt": pl.UInt32,
    }


def _candidates_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "order_index": pl.UInt32,
        "flickr_photo_id": pl.String,
        "class_kind": pl.String,
        "class_id": pl.String,
        "accepted_taxon_key": pl.String,
        "display_name": pl.String,
        "text_prompt": pl.String,
        "candidate_reason": pl.String,
        "target_candidate": pl.Boolean,
        "text_similarity": pl.Float64,
        "reference_prototype_similarity": pl.Float64,
        "prototype_score": pl.Float64,
        "rank": pl.UInt32,
        "score_semantics": pl.String,
        "experimental_screening_evidence": pl.Boolean,
    }


def _failures_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "order_index": pl.UInt32,
        "flickr_photo_id": pl.String,
        "failure_stage": pl.String,
        "retryable": pl.Boolean,
        "error_type": pl.String,
        "error": pl.String,
        "biological_negative": pl.Boolean,
        "occurred_at": pl.String,
        "attempt": pl.UInt32,
    }


def _bioclip_scorer(config: PrototypeStagedFlickrConfig) -> PersistentBioClipScorer:
    runtime = BioClipRuntime(
        model=ModelConfig(
            model_id="bioclip2_5_huge",
            display_name="BioCLIP 2.5 Huge",
            role="preferred",
            status="use_if_available",
            task="staged target-aware Flickr prototype",
            model_name=config.model_name,
            checkpoint=config.model_revision,
            package_name="open_clip_torch",
            package_version=config.open_clip_version,
            model_hash=f"hf-revision:{config.model_revision}",
        ),
        home=config.bioclip_runtime_python.parent.parent,
        venv_python=_absolute(config.bioclip_runtime_python),
        package_version=config.open_clip_version,
        available=True,
    )
    return PersistentBioClipScorer(
        runtime=runtime,
        hf_cache_dir=config.bioclip_hf_cache_dir,
        device=config.device,
        image_resize_mode="longest",
        preprocess_workers=config.preprocess_workers,
    )


def _yoloe_detector(
    config: PrototypeStagedFlickrConfig,
) -> YoloE26SidecarObjectDetector:
    return YoloE26SidecarObjectDetector(
        runtime_python=str(_absolute(config.yoloe_runtime_python)),
        checkpoint=config.yoloe_checkpoint,
        device=config.device,
        imgsz=config.yoloe_imgsz,
        conf=config.yoloe_conf,
        iou=config.yoloe_iou,
        max_det=config.yoloe_max_det,
        transport="json_b64",
    )


def _model_report(
    scorer: ImageEmbeddingScorer,
    config: PrototypeStagedFlickrConfig,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "model_id": config.model_name,
        "model_revision": config.model_revision,
        "model_weights_sha256": scorer.model_weights_sha256,
        "open_clip_version": config.open_clip_version,
        "open_clip_config_sha256": scorer.open_clip_config_sha256,
        "preprocessing_version": scorer.preprocessing_version,
        "preprocessing_fingerprint": scorer.preprocessing_fingerprint,
        "image_resize_mode": scorer.effective_image_resize_mode,
        "device_requested": config.device,
        "device_actual": scorer.device,
        "gpu_name": scorer.gpu_name,
        "persistent_loading": dict(scorer.cache_metrics),
        "memory": dict(scorer.memory_metrics),
        "runtime_evidence_scope": "current_invocation_attestation",
        "full_stream_reuse_verified_by_stage_gates": True,
        "elapsed_seconds": round(elapsed_seconds, 6),
    }


def _detector_report(detector: RouteDetector) -> dict[str, object]:
    return {
        "model_id": detector.model_id,
        "model_version": detector.model_version,
        "checkpoint": detector.checkpoint,
        "prompt_set_fingerprint": detector.prompt_set_fingerprint,
        "persistent_worker_process_starts": detector.worker_process_starts,
        "persistent_worker_requests": detector.worker_request_count,
        "role": "gate_and_route_classifier_only",
        "runtime_evidence_scope": "current_invocation_attestation",
        "full_stream_reuse_verified_by_stage_gates": True,
    }


def _write_progress(
    path: Path,
    *,
    stage_number: int,
    stage_limit: int,
    counts: Mapping[str, int],
    elapsed_seconds: float,
) -> None:
    payload = {
        "schema_version": STAGED_FLICKR_REPORT_VERSION,
        "status": "running",
        "stage_id": f"P{stage_number}",
        "stage_limit": stage_limit,
        "counts": dict(counts),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "updated_at": _utc_now(),
    }
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_previous_report(path: Path) -> Mapping[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping):
        return None
    if value.get("schema_version") != STAGED_FLICKR_REPORT_VERSION:
        return None
    return value


def _previous_stage(
    report: Mapping[str, object] | None,
    *,
    stage_number: int,
    stage_limit: int,
) -> dict[str, object]:
    if report is None or not isinstance(report.get("stages"), Sequence):
        raise RuntimeError(
            f"stage P{stage_number} is checkpoint-complete but prior gate evidence is missing"
        )
    for value in report["stages"]:
        if not isinstance(value, Mapping):
            continue
        if (
            value.get("stage_id") == f"P{stage_number}"
            and value.get("cumulative_limit") == stage_limit
            and value.get("status") == "passed"
        ):
            return dict(value)
    raise RuntimeError(
        f"stage P{stage_number} is checkpoint-complete but prior gate evidence is invalid"
    )


def _summary(report: Mapping[str, object]) -> str:
    counts = report["counts"]
    assert isinstance(counts, Mapping)
    lines = [
        "# Staged target-aware Flickr prototype",
        "",
        f"Status: `{report['status']}`.",
        f"Classified: {counts['classified']} of {counts['planned']} planned records.",
        f"Operational failures: {counts['failures']}.",
        "Storage: local only; S3 was not accessed.",
        "",
        "All outputs are experimental screening evidence. Scores and margins are uncalibrated and are not probabilities or verified biological identifications.",
        "",
        "## Gates",
        "",
    ]
    stages = report["stages"]
    assert isinstance(stages, Sequence)
    for stage in stages:
        assert isinstance(stage, Mapping)
        lines.append(
            f"- {stage['stage_id']}: {stage['classified']} classified, "
            f"{stage['failures']} failures, {stage['records_per_second']} records/s."
        )
    return "\n".join(lines) + "\n"


def _artifact(path: Path) -> dict[str, object]:
    return {
        "uri": str(path),
        "byte_count": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _remove_cache(cache_dir: Path) -> None:
    if cache_dir.exists():
        for path in cache_dir.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)
        try:
            cache_dir.rmdir()
            cache_dir.parent.rmdir()
        except OSError:
            pass


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _maxrss_bytes(who: int) -> int:
    value = int(resource.getrusage(who).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _required_text(value: object, *, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field} must be non-empty")
    return result


def _require_sha256(value: object, *, field: str) -> str:
    result = str(value)
    if len(result) != 71 or not result.startswith("sha256:"):
        raise ValueError(f"{field} must be a SHA-256 fingerprint")
    try:
        int(result.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a SHA-256 fingerprint") from exc
    return result
