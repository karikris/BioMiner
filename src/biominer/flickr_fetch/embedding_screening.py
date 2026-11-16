"""Bounded, resumable local full-frame embedding evidence for a Flickr snapshot.

This developer workflow produces provisional route suggestions. It does not
measure organisms, admit references, or make occurrence release decisions.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Callable, Protocol, Sequence
from urllib.request import Request, urlopen

from PIL import Image
import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.detection.detector_base import DecodedImage
from biominer.detection.embeddinggemma2 import (
    EmbeddingGemma2Config,
    EmbeddingGemma2SidecarRouter,
)
from biominer.storage.parquet import ParquetRowSource, write_parquet_batches


SCREENING_SCHEMA_VERSION = "flickr-embedding-screening-v1"
SOURCE_COLUMNS = (
    "source", "flickr_photo_id", "image_url", "image_url_kind",
    "source_record_hash", "query_term", "query_language", "query_field",
    "source_created_at",
)
REGISTER_SCHEMA = {**dict.fromkeys(SOURCE_COLUMNS, pl.String),
                   "sample_rank": pl.Int64, "shard_index": pl.Int64}
ROUTE_SCHEMA = {
    **REGISTER_SCHEMA,
    "schema_version": pl.String, "model_id": pl.String,
    "model_version": pl.String, "model_revision": pl.String,
    "model_fingerprint": pl.String, "prompt_set_fingerprint": pl.String,
    "image_content_hash": pl.String, "embedding": pl.List(pl.Float32),
    "proposed_domain": pl.String, "bioclip_route": pl.String,
    "routing_action": pl.String, "routing_reason": pl.String,
    "cosine_similarity": pl.Float64, "similarity_margin": pl.Float64,
    "prompt_similarities_json": pl.String, "evidence_kind": pl.String,
    "localization_status": pl.String, "subject_present": pl.Boolean,
    "subject_area_ratio": pl.Float64, "detection_count": pl.Int64,
    "bbox_xyxy": pl.List(pl.Float64),
    "mask_polygon_xyn": pl.List(pl.List(pl.Float64)), "detector_score": pl.Float64,
}
FAILURE_SCHEMA = {"source": pl.String, "flickr_photo_id": pl.String,
                  "attempts": pl.Int64, "stage": pl.String, "error": pl.String}


class Router(Protocol):
    model_id: str
    model_version: str
    fingerprint: str
    prompt_set_fingerprint: str

    def route_batch(self, images: Sequence[DecodedImage]) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


class _RouterBatchFailure(RuntimeError):
    """The shard coordinator must replace the failed persistent runtime."""


@dataclass(frozen=True, slots=True)
class EmbeddingScreeningConfig:
    state_db: Path
    output_dir: Path
    reports_dir: Path
    model: EmbeddingGemma2Config = field(default_factory=EmbeddingGemma2Config)
    runtime_python: Path = field(default_factory=lambda: Path(sys.executable))
    sample_size: int | None = 10_000
    sample_seed: str = "flickr-embeddinggemma2-v1"
    shard_count: int = 1
    expected_images: int | None = None
    download_workers: int = 4
    batch_size: int = 8
    timeout_seconds: float = 30.0
    retries: int = 2
    max_image_bytes: int = 20_000_000
    max_attempts: int = 3

    def __post_init__(self) -> None:
        for key in ("state_db", "output_dir", "reports_dir", "runtime_python"):
            object.__setattr__(self, key, Path(getattr(self, key)).expanduser())
        for key in ("shard_count", "download_workers", "batch_size", "max_image_bytes", "max_attempts"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.sample_size is not None and (isinstance(self.sample_size, bool)
                or not isinstance(self.sample_size, int) or self.sample_size <= 0):
            raise ValueError("sample_size must be a positive integer or None")
        if self.expected_images is not None and (isinstance(self.expected_images, bool)
                or not isinstance(self.expected_images, int) or self.expected_images <= 0):
            raise ValueError("expected_images must be a positive integer or None")
        if not self.sample_seed.strip():
            raise ValueError("sample_seed must not be empty")
        if self.retries < 0 or self.timeout_seconds <= 0:
            raise ValueError("retries must be nonnegative and timeout_seconds positive")


@dataclass(frozen=True, slots=True)
class EmbeddingScreeningResult:
    manifest_path: Path
    sample_path: Path
    results_path: Path
    failures_path: Path
    report_path: Path
    report: dict[str, Any]


def run_embedding_screening(
    config: EmbeddingScreeningConfig,
    *,
    router_factory: Callable[[EmbeddingGemma2Config], Router] | None = None,
) -> EmbeddingScreeningResult:
    """Use one persistent router per shard; publish only terminal outcomes."""
    with _local_output_lock(config.output_dir):
        return _run_embedding_screening(config, router_factory=router_factory)


@contextmanager
def _local_output_lock(directory: Path):
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".embedding_screening.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("embedding screening output is locked by an active writer") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _run_embedding_screening(config, *, router_factory):
    snapshot, receipt = _prepare_snapshot(config)
    factory = router_factory or (lambda model: EmbeddingGemma2SidecarRouter(
        runtime_python=str(config.runtime_python), config=model))
    with ThreadPoolExecutor(max_workers=config.shard_count) as pool:
        futures = [pool.submit(_run_shard, config, snapshot, receipt, index, factory)
                   for index in range(config.shard_count)]
        shards = [future.result() for future in futures]
    if len({item["runtime_fingerprint"] for item in shards}) != 1:
        raise ValueError("shards used different runtime identities; refusing publication")
    routes_path = config.output_dir / "image_level_routes.parquet"
    failures_path = config.output_dir / "failures.parquet"
    write_parquet_batches(
        (batch for shard in shards for batch in _state_batches(Path(shard["state_db"]), "results")),
        routes_path, schema=ROUTE_SCHEMA,
    )
    write_parquet_batches(
        (batch for shard in shards for batch in _state_batches(Path(shard["state_db"]), "failures")),
        failures_path, schema=FAILURE_SCHEMA,
    )
    image_routes = pl.scan_parquet(routes_path).select(pl.len()).collect().item()
    failures = pl.scan_parquet(failures_path).select(pl.len()).collect().item()
    if image_routes + failures != receipt["selected_records"]:
        raise ValueError("shard outputs do not cover the immutable snapshot")
    domains = dict(pl.scan_parquet(routes_path).group_by("proposed_domain").len()
                   .collect().iter_rows()) if image_routes else {}
    report = {
        "schema_version": SCREENING_SCHEMA_VERSION,
        "status": "complete" if failures == 0 else "complete_with_operational_failures",
        "scientific_scope": "Provisional full-frame embedding evidence; not taxonomic validation, "
                            "organism counts, reference admission, or occurrence release.",
        "counts": {"image_routes": image_routes, "terminal_operational_failures": failures,
                   "domains": dict(sorted(domains.items()))},
        "identity": receipt["identity"], "snapshot_sha256": receipt["snapshot_sha256"],
        "runtime_fingerprint": shards[0]["runtime_fingerprint"],
        "shards": shards, "generated_at": datetime.now(UTC).isoformat(),
        "artifacts": {"routes_sha256": _file_hash(routes_path),
                      "failures_sha256": _file_hash(failures_path)},
        "localization_status": "unavailable", "reference_admission": "review_required",
        "occurrence_release": "not_assessed",
    }
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.reports_dir / f"embeddinggemma2_flickr_{config.output_dir.name}.json"
    _write_json(report_path, report)
    manifest_path = config.output_dir / "embedding_screening_manifest.json"
    _write_json(manifest_path, report)  # Coordinator publishes the completion manifest last.
    return EmbeddingScreeningResult(manifest_path, snapshot, routes_path, failures_path, report_path, report)


def _identity(config: EmbeddingScreeningConfig) -> dict[str, Any]:
    return {
        "schema_version": SCREENING_SCHEMA_VERSION,
        "source_db": str(config.state_db.resolve()), "model": config.model.as_dict(),
        "runtime_python": str(config.runtime_python.absolute()),
        "sample_size": config.sample_size, "sample_seed": config.sample_seed,
        "expected_images": config.expected_images,
        "shard_count": config.shard_count, "download_workers": config.download_workers,
        "batch_size": config.batch_size, "max_attempts": config.max_attempts,
        "timeout_seconds": config.timeout_seconds, "retries": config.retries,
        "max_image_bytes": config.max_image_bytes,
    }


def _prepare_snapshot(config: EmbeddingScreeningConfig) -> tuple[Path, dict[str, Any]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = config.output_dir / "sample_register.parquet"
    manifest = config.output_dir / "snapshot_manifest.json"
    identity = _identity(config)
    if manifest.exists():
        receipt = json.loads(manifest.read_text())
        if receipt.get("identity") != identity:
            raise ValueError("snapshot manifest conflicts with requested identity")
        if not snapshot.is_file() or _file_hash(snapshot) != receipt.get("snapshot_sha256"):
            raise ValueError("snapshot checksum mismatch")
        return snapshot, receipt
    if any(path.name != ".embedding_screening.lock" for path in config.output_dir.iterdir()):
        raise ValueError("output directory contains unbound or legacy state; use an empty directory")
    with sqlite3.connect(config.state_db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.execute("BEGIN")
        population = conn.execute("SELECT COUNT(*) FROM source_records WHERE image_url != ''").fetchone()[0]
        if config.expected_images is not None and population != config.expected_images:
            raise ValueError(f"source population {population} differs from expected_images {config.expected_images}")
        selected = population if config.sample_size is None else config.sample_size
        if selected > population or selected < config.shard_count:
            raise ValueError("source population cannot satisfy sample size and shard count")
        duplicate = conn.execute("""SELECT 1 FROM source_records WHERE image_url != ''
            GROUP BY source, flickr_photo_id HAVING COUNT(*) > 1 LIMIT 1""").fetchone()
        if duplicate:
            raise ValueError("source snapshot contains duplicate source/photo identities")
        conn.create_function("screen_key", 2, lambda source, photo:
            sha256(f"{config.sample_seed}:{source}:{photo}".encode()).hexdigest(), deterministic=True)
        cursor = conn.execute("""SELECT source, flickr_photo_id, image_url, image_url_kind,
            source_record_hash, query_term, query_language, query_field, created_at
            FROM source_records WHERE image_url != ''
            ORDER BY screen_key(source, flickr_photo_id), source, flickr_photo_id LIMIT ?""", (selected,))

        def batches():
            rank = 0
            while values := cursor.fetchmany(1_000):
                rows = []
                for value in values:
                    row = dict(zip(SOURCE_COLUMNS, value, strict=True))
                    if not row["source"] or not row["flickr_photo_id"] or not row["source_record_hash"]:
                        raise ValueError("snapshot records require source, photo ID, and source hash")
                    rank += 1
                    rows.append({**row, "sample_rank": rank,
                                 "shard_index": (rank - 1) % config.shard_count})
                yield pl.DataFrame(rows, schema=REGISTER_SCHEMA)

        write_parquet_batches(batches(), snapshot, schema=REGISTER_SCHEMA)
    receipt = {"identity": identity, "population": population, "selected_records": selected,
               "snapshot_sha256": _file_hash(snapshot)}
    _write_json(manifest, receipt)
    return snapshot, receipt


def _run_shard(config, snapshot, receipt, index, factory):
    directory = config.output_dir / f"shard_{index:03d}"
    directory.mkdir(exist_ok=True)
    state = directory / "embedding_state.sqlite"
    router = factory(config.model)
    restarts = 0
    try:
        runtime_fingerprint = router.fingerprint
        state_identity = canonical_semantic_fingerprint({
            "snapshot": receipt["snapshot_sha256"], "identity": receipt["identity"],
            "runtime_fingerprint": runtime_fingerprint, "shard_index": index,
        })
        _init_state(state, state_identity)
        while True:
            pending = []
            try:
                with sqlite3.connect(state) as conn, ThreadPoolExecutor(max_workers=config.download_workers) as downloads:
                    for row in ParquetRowSource(snapshot):
                        if row["shard_index"] != index:
                            continue
                        key = (row["source"], row["flickr_photo_id"])
                        if conn.execute("SELECT 1 FROM results WHERE source=? AND photo=?", key).fetchone():
                            continue
                        failure = conn.execute("SELECT attempts FROM failures WHERE source=? AND photo=?", key).fetchone()
                        if failure and failure[0] >= config.max_attempts:
                            continue
                        pending.append(row)
                        if len(pending) == config.batch_size:
                            _run_batch(pending, conn, config, router, downloads)
                            pending = []
                    if pending:
                        _run_batch(pending, conn, config, router, downloads)
                    count = conn.execute("SELECT COUNT(*) FROM failures WHERE attempts < ?", (config.max_attempts,)).fetchone()[0]
            except _RouterBatchFailure as exc:
                router.close()
                restarts += 1
                if restarts > config.max_attempts:
                    raise RuntimeError("embedding runtime restart budget exhausted; checkpoints preserved") from exc
                router = factory(config.model)
                if router.fingerprint != runtime_fingerprint:
                    raise ValueError("replacement runtime identity mismatch")
                continue
            if count == 0:
                break
        return {"shard_index": index, "state_db": str(state),
                "runtime_fingerprint": runtime_fingerprint}
    finally:
        router.close()


def _init_state(path: Path, identity: str) -> None:
    with sqlite3.connect(path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables and "metadata" not in tables:
            raise ValueError("unbound legacy runtime state")
        conn.execute("CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL)")
        current = conn.execute("SELECT identity FROM metadata").fetchall()
        if current and current != [(identity,)]:
            raise ValueError("runtime identity mismatch; use a new output directory")
        if not current:
            if tables:
                raise ValueError("unbound runtime identity")
            conn.execute("INSERT INTO metadata VALUES (?)", (identity,))
        conn.execute("""CREATE TABLE IF NOT EXISTS results (
            source TEXT, photo TEXT, result_json TEXT NOT NULL, PRIMARY KEY (source, photo))""")
        conn.execute("""CREATE TABLE IF NOT EXISTS failures (
            source TEXT, photo TEXT, attempts INTEGER NOT NULL, stage TEXT, error TEXT,
            PRIMARY KEY (source, photo))""")


def _run_batch(rows, conn, config, router, downloads):
    jobs = [downloads.submit(_download_decode, row, config) for row in rows]
    loaded = []
    for row, job in zip(rows, jobs, strict=True):
        try:
            loaded.append((row, job.result()))
        except Exception as exc:
            _failure(conn, row, "download_or_decode", exc)
    if loaded:
        try:
            routes = router.route_batch([image for _, image in loaded])
            if len(routes) != len(loaded):
                raise ValueError("embedding router returned a different number of image rows")
            prepared = []
            for (row, _), evidence in zip(loaded, routes, strict=True):
                if evidence.get("routing_action") != "review":
                    raise ValueError("unvalidated embedding routes require review")
                for name in ("subject_present", "subject_area_ratio", "detection_count", "bbox_xyxy", "mask_polygon_xyn", "detector_score"):
                    if evidence.get(name) is not None:
                        raise ValueError(f"embedding router cannot supply measured {name}")
                value = {**evidence, **row, "schema_version": SCREENING_SCHEMA_VERSION,
                         "model_id": router.model_id, "model_version": router.model_version,
                         "model_revision": config.model.revision,
                         "model_fingerprint": router.fingerprint,
                         "prompt_set_fingerprint": router.prompt_set_fingerprint}
                prepared.append((row["source"], row["flickr_photo_id"], json.dumps(value, allow_nan=False)))
            conn.executemany("INSERT INTO results VALUES (?, ?, ?)", prepared)
            conn.executemany("DELETE FROM failures WHERE source=? AND photo=?", [(row["source"], row["flickr_photo_id"]) for row, _ in loaded])
        except Exception as exc:
            for row, _ in loaded:
                _failure(conn, row, "embedding_inference", exc)
            conn.commit()
            raise _RouterBatchFailure(str(exc)) from exc
    conn.commit()


def _failure(conn, row, stage, error):
    conn.execute("""INSERT INTO failures VALUES (?, ?, 1, ?, ?)
        ON CONFLICT(source, photo) DO UPDATE SET attempts=attempts+1,
        stage=excluded.stage, error=excluded.error""",
        (row["source"], row["flickr_photo_id"], stage, f"{type(error).__name__}: {error}"))


def _state_batches(path: Path, table: str):
    query = ("SELECT result_json FROM results ORDER BY source, photo" if table == "results"
             else "SELECT source, photo, attempts, stage, error FROM failures ORDER BY source, photo")
    with sqlite3.connect(path) as conn:
        cursor = conn.execute(query)
        while values := cursor.fetchmany(1_000):
            rows = ([json.loads(item[0]) for item in values] if table == "results"
                    else [dict(zip(FAILURE_SCHEMA, item, strict=True)) for item in values])
            yield pl.DataFrame(rows, schema=ROUTE_SCHEMA if table == "results" else FAILURE_SCHEMA)


def _download_decode(row, config) -> DecodedImage:
    request = Request(row["image_url"], headers={"User-Agent": "BioMiner/0.1 embedding screening"})
    last = None
    for _ in range(config.retries + 1):
        try:
            with urlopen(request, timeout=config.timeout_seconds) as response:
                if not str(response.headers.get("Content-Type", "")).lower().startswith("image/"):
                    raise ValueError("download is not an image")
                payload = response.read(config.max_image_bytes + 1)
            if len(payload) > config.max_image_bytes:
                raise ValueError("image exceeds max_image_bytes")
            with Image.open(BytesIO(payload)) as image:
                rgb = image.convert("RGB")
                return DecodedImage(rgb.width, rgb.height, "RGB", rgb.tobytes(), source_uri=row["image_url"])
        except Exception as exc:
            last = exc
    raise OSError(f"image download/decode failed: {type(last).__name__}: {last}")


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
