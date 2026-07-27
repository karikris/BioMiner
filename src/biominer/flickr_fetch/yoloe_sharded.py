"""Resumable multi-model YOLOE screening of a canonical Flickr snapshot."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any

import polars as pl

from biominer.detection.yoloe26_detector import (
    YoloE26SidecarObjectDetector,
    yoloe26_prompt_set_fingerprint,
)
from biominer.flickr_fetch.yoloe_pilot import (
    YoloeBatchError,
    YoloePilotConfig,
    run_yoloe_pilot,
)
from biominer.storage.parquet import write_parquet


SHARDED_RUN_VERSION = "flickr-yoloe-sharded-v1"
SOURCE_COLUMNS = (
    "source",
    "flickr_photo_id",
    "image_url",
    "image_url_kind",
    "source_record_hash",
    "query_term",
    "query_language",
    "query_field",
    "source_created_at",
)


@dataclass(frozen=True, slots=True)
class ShardedYoloeConfig:
    state_db: Path
    output_dir: Path
    reports_dir: Path
    expected_images: int | None = None
    shard_count: int = 4
    sample_seed: str = "australia-flickr-yoloe-full-v1"
    runtime_python: Path = Path("../YOLO26/venv/bin/python")
    checkpoint: str = "yoloe-26s-seg.pt"
    device: str = "mps"
    imgsz: int = 768
    confidence: float = 0.20
    iou: float = 0.50
    max_det: int = 8
    download_workers_per_shard: int = 4
    yoloe_batch_size: int = 8
    timeout_seconds: float = 30.0
    retries: int = 2
    max_image_bytes: int = 20_000_000
    max_attempts: int = 3
    prompt_classes: tuple[str, ...] = ("insect",)

    def __post_init__(self) -> None:
        if self.expected_images is not None and self.expected_images <= 0:
            raise ValueError("expected_images must be positive when provided")
        if self.shard_count <= 0:
            raise ValueError("shard_count must be positive")
        if self.download_workers_per_shard <= 0 or self.yoloe_batch_size <= 0:
            raise ValueError("download workers and YOLOE batch size must be positive")
        if self.max_image_bytes <= 0 or self.max_attempts <= 0:
            raise ValueError("max_image_bytes and max_attempts must be positive")
        if self.device not in {"mps", "cpu", "auto"}:
            raise ValueError("device must be mps, cpu, or auto")
        if not self.prompt_classes:
            raise ValueError("prompt_classes must contain at least one prompt")


@dataclass(frozen=True, slots=True)
class PreparedYoloeShards:
    manifest_path: Path
    sample_path: Path
    population: int
    shard_paths: tuple[Path, ...]
    shard_sizes: tuple[int, ...]
    source_snapshot_fingerprint: str


@dataclass(frozen=True, slots=True)
class ShardedYoloeResult:
    manifest_path: Path
    sample_path: Path
    results_path: Path
    failures_path: Path
    report_path: Path
    summary_path: Path
    report: dict[str, Any]


def prepare_yoloe_shards(config: ShardedYoloeConfig) -> PreparedYoloeShards:
    """Materialize or verify one immutable, balanced source snapshot."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "shard_manifest.json"
    sample_path = config.output_dir / "sample_register.parquet"
    if manifest_path.exists():
        return _load_prepared_shards(config, manifest_path, sample_path)

    if any(config.output_dir.glob("shard_*/pilot_state.sqlite")):
        raise ValueError("shard state exists without an immutable shard manifest")
    source_rows = _read_source_rows(config.state_db)
    population = len(source_rows)
    if config.expected_images is not None and population != config.expected_images:
        raise ValueError(
            f"source snapshot has {population} unique image records; expected {config.expected_images}"
        )
    if population < config.shard_count:
        raise ValueError("source snapshot has fewer records than requested shards")

    ordered = sorted(
        source_rows,
        key=lambda row: (
            sha256(
                f"{config.sample_seed}:{row['source']}:{row['flickr_photo_id']}".encode()
            ).digest(),
            str(row["source"]),
            str(row["flickr_photo_id"]),
        ),
    )
    source_fingerprint = _source_snapshot_fingerprint(ordered)
    shard_rows: list[list[dict[str, Any]]] = [[] for _ in range(config.shard_count)]
    combined_rows: list[dict[str, Any]] = []
    for global_rank, row in enumerate(ordered, start=1):
        shard_index = (global_rank - 1) % config.shard_count
        routed = {
            **row,
            "sample_rank": len(shard_rows[shard_index]) + 1,
            "global_sample_rank": global_rank,
            "shard_index": shard_index,
            "snapshot_population_with_image_url": population,
        }
        shard_rows[shard_index].append(routed)
        combined_rows.append(routed)

    write_parquet(pl.DataFrame(combined_rows), sample_path)
    shard_paths: list[Path] = []
    for shard_index, rows in enumerate(shard_rows):
        shard_path = config.output_dir / f"shard_{shard_index:03d}"
        shard_path.mkdir(parents=True, exist_ok=True)
        write_parquet(pl.DataFrame(rows), shard_path / "sample_register.parquet")
        shard_paths.append(shard_path)

    manifest = _manifest_payload(
        config,
        population=population,
        source_fingerprint=source_fingerprint,
        shard_sizes=tuple(len(rows) for rows in shard_rows),
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return PreparedYoloeShards(
        manifest_path=manifest_path,
        sample_path=sample_path,
        population=population,
        shard_paths=tuple(shard_paths),
        shard_sizes=tuple(len(rows) for rows in shard_rows),
        source_snapshot_fingerprint=source_fingerprint,
    )


def run_sharded_yoloe(config: ShardedYoloeConfig) -> ShardedYoloeResult:
    """Run one persistent sidecar per shard and merge only after all are terminal."""
    started_at = _now()
    prepared = prepare_yoloe_shards(config)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    shard_reports: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.shard_count) as pool:
        futures = {
            pool.submit(_run_one_shard, config, shard_index, shard_path, shard_size): shard_index
            for shard_index, (shard_path, shard_size) in enumerate(
                zip(prepared.shard_paths, prepared.shard_sizes, strict=True)
            )
        }
        for future in as_completed(futures):
            shard_reports.append(future.result())
    shard_reports.sort(key=lambda item: int(item["shard_index"]))

    results, failures = _merge_shard_outputs(config, prepared, shard_reports)
    results_path = write_parquet(results, config.output_dir / "image_level_routes.parquet")
    failures_path = write_parquet(failures, config.output_dir / "failures.parquet")
    report = _combined_report(
        config,
        prepared,
        shard_reports,
        results,
        failures,
        started_at=started_at,
    )
    report_path = config.reports_dir / f"yoloe_flickr_sharded_{config.output_dir.name}.json"
    summary_path = report_path.with_suffix(".md")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(_markdown_summary(report), encoding="utf-8")
    return ShardedYoloeResult(
        manifest_path=prepared.manifest_path,
        sample_path=prepared.sample_path,
        results_path=results_path,
        failures_path=failures_path,
        report_path=report_path,
        summary_path=summary_path,
        report=report,
    )


def _run_one_shard(
    config: ShardedYoloeConfig,
    shard_index: int,
    shard_path: Path,
    shard_size: int,
) -> dict[str, Any]:
    pilot_config = YoloePilotConfig(
        state_db=config.state_db,
        output_dir=shard_path,
        reports_dir=config.reports_dir,
        sample_size=shard_size,
        sample_seed=config.sample_seed,
        runtime_python=config.runtime_python,
        checkpoint=config.checkpoint,
        device=config.device,
        imgsz=config.imgsz,
        confidence=config.confidence,
        iou=config.iou,
        max_det=config.max_det,
        download_workers=config.download_workers_per_shard,
        yoloe_batch_size=config.yoloe_batch_size,
        timeout_seconds=config.timeout_seconds,
        retries=config.retries,
        max_image_bytes=config.max_image_bytes,
        max_attempts=config.max_attempts,
        prompt_classes=config.prompt_classes,
    )
    detector = _new_detector(config)
    previous_progress: tuple[int, int, int] | None = None
    try:
        while True:
            try:
                result = run_yoloe_pilot(pilot_config, detector=detector)
            except YoloeBatchError:
                detector.close()
                detector = _new_detector(config)
                continue
            state = _shard_state(shard_path / "pilot_state.sqlite", shard_size, config.max_attempts)
            if state["classified"] + state["terminal_failures"] == shard_size:
                return {
                    "shard_index": shard_index,
                    "shard_size": shard_size,
                    "status": "complete" if not state["terminal_failures"] else "complete_with_failures",
                    "classified": state["classified"],
                    "terminal_failures": state["terminal_failures"],
                    "attempts": state["attempts"],
                    "result_parquet": str(result.results_path),
                    "failure_parquet": str(result.failures_path),
                    "state_db": str(shard_path / "pilot_state.sqlite"),
                }
            progress = (state["classified"], state["terminal_failures"], state["attempts"])
            if progress == previous_progress:
                raise RuntimeError(f"YOLOE shard {shard_index} made no progress and remains incomplete")
            previous_progress = progress
    finally:
        detector.close()


def _new_detector(config: ShardedYoloeConfig) -> YoloE26SidecarObjectDetector:
    return YoloE26SidecarObjectDetector(
        runtime_python=str(config.runtime_python),
        checkpoint=config.checkpoint,
        device=config.device,
        imgsz=config.imgsz,
        conf=config.confidence,
        iou=config.iou,
        max_det=config.max_det,
        prompt_classes=config.prompt_classes,
    )


def _read_source_rows(path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            """
            SELECT source, flickr_photo_id, image_url, image_url_kind, source_record_hash,
                   query_term, query_language, query_field, created_at
            FROM source_records WHERE image_url != '' ORDER BY source, flickr_photo_id
            """
        ).fetchall()
    records = [dict(zip(SOURCE_COLUMNS, row, strict=True)) for row in rows]
    identities = {(str(row["source"]), str(row["flickr_photo_id"])) for row in records}
    if len(identities) != len(records):
        raise ValueError("source snapshot contains duplicate source/photo identities")
    return records


def _source_snapshot_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = sha256()
    for row in rows:
        payload = [row.get(column) for column in SOURCE_COLUMNS]
        digest.update(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode())
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def _manifest_payload(
    config: ShardedYoloeConfig,
    *,
    population: int,
    source_fingerprint: str,
    shard_sizes: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "schema_version": SHARDED_RUN_VERSION,
        "created_at": _now(),
        "state_db": str(config.state_db.expanduser().absolute()),
        "population": population,
        "sample_seed": config.sample_seed,
        "source_snapshot_fingerprint": source_fingerprint,
        "shard_count": config.shard_count,
        "shard_sizes": list(shard_sizes),
        "partition": "sha256(seed:source:flickr_photo_id), sorted, zero-based rank modulo shard_count",
        "runtime": {
            "runtime_python": str(config.runtime_python.expanduser().absolute()),
            "checkpoint": config.checkpoint,
            "device": config.device,
            "imgsz": config.imgsz,
            "confidence": config.confidence,
            "iou": config.iou,
            "max_det": config.max_det,
            "download_workers_per_shard": config.download_workers_per_shard,
            "total_download_workers": config.download_workers_per_shard * config.shard_count,
            "yoloe_batch_size": config.yoloe_batch_size,
            "prompt_classes": list(config.prompt_classes),
            "prompt_set_fingerprint": yoloe26_prompt_set_fingerprint(config.prompt_classes),
        },
    }


def _load_prepared_shards(
    config: ShardedYoloeConfig,
    manifest_path: Path,
    sample_path: Path,
) -> PreparedYoloeShards:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_identity = _manifest_payload(
        config,
        population=int(manifest.get("population", 0)),
        source_fingerprint=str(manifest.get("source_snapshot_fingerprint", "")),
        shard_sizes=tuple(int(item) for item in manifest.get("shard_sizes", [])),
    )
    for key in (
        "schema_version",
        "state_db",
        "population",
        "sample_seed",
        "source_snapshot_fingerprint",
        "shard_count",
        "shard_sizes",
        "partition",
        "runtime",
    ):
        if manifest.get(key) != expected_identity.get(key):
            raise ValueError(f"existing shard manifest conflicts with requested {key}")
    population = int(manifest["population"])
    if config.expected_images is not None and population != config.expected_images:
        raise ValueError(
            f"existing shard snapshot has {population} records; expected {config.expected_images}"
        )
    shard_sizes = tuple(int(item) for item in manifest["shard_sizes"])
    shard_paths = tuple(config.output_dir / f"shard_{index:03d}" for index in range(config.shard_count))
    if not sample_path.exists() or pl.scan_parquet(sample_path).select(pl.len()).collect().item() != population:
        raise ValueError("combined sample register is missing or has the wrong row count")
    for shard_path, shard_size in zip(shard_paths, shard_sizes, strict=True):
        register = shard_path / "sample_register.parquet"
        if not register.exists() or pl.scan_parquet(register).select(pl.len()).collect().item() != shard_size:
            raise ValueError(f"shard register is missing or invalid: {register}")
    return PreparedYoloeShards(
        manifest_path=manifest_path,
        sample_path=sample_path,
        population=population,
        shard_paths=shard_paths,
        shard_sizes=shard_sizes,
        source_snapshot_fingerprint=str(manifest["source_snapshot_fingerprint"]),
    )


def _shard_state(path: Path, shard_size: int, max_attempts: int) -> dict[str, int]:
    with sqlite3.connect(path) as conn:
        classified = int(conn.execute("SELECT COUNT(*) FROM results").fetchone()[0])
        failure_row = conn.execute(
            """
            SELECT COALESCE(SUM(f.attempts), 0),
                   COALESCE(SUM(CASE WHEN f.attempts >= ? THEN 1 ELSE 0 END), 0)
            FROM failures AS f
            LEFT JOIN results AS r USING (flickr_photo_id)
            WHERE r.flickr_photo_id IS NULL
            """,
            (max_attempts,),
        ).fetchone()
    terminal = int(failure_row[1])
    if classified + terminal > shard_size:
        raise ValueError("shard state contains more terminal records than its register")
    return {"classified": classified, "terminal_failures": terminal, "attempts": int(failure_row[0])}


def _merge_shard_outputs(
    config: ShardedYoloeConfig,
    prepared: PreparedYoloeShards,
    shard_reports: list[dict[str, Any]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if sum(int(item["shard_size"]) for item in shard_reports) != prepared.population:
        raise ValueError("shard reports do not cover the immutable source snapshot")
    result_frames = [pl.read_parquet(item["result_parquet"]) for item in shard_reports]
    failure_frames = [pl.read_parquet(item["failure_parquet"]) for item in shard_reports]
    results = pl.concat(result_frames, how="diagonal_relaxed").sort("flickr_photo_id")
    failures = pl.concat(failure_frames, how="diagonal_relaxed")
    if not failures.is_empty():
        failures = failures.filter(~pl.col("flickr_photo_id").is_in(results["flickr_photo_id"])).sort(
            "flickr_photo_id"
        )
    if results["flickr_photo_id"].n_unique() != results.height:
        raise ValueError("shard result Parquets overlap")
    if results.height + failures.height != prepared.population:
        raise ValueError("shard outputs are incomplete; refusing to publish merged Parquets")
    if not failures.is_empty() and int(failures["attempts"].min()) < config.max_attempts:
        raise ValueError("retryable failures remain; refusing to publish merged Parquets")
    return results, failures


def _combined_report(
    config: ShardedYoloeConfig,
    prepared: PreparedYoloeShards,
    shard_reports: list[dict[str, Any]],
    results: pl.DataFrame,
    failures: pl.DataFrame,
    *,
    started_at: str,
) -> dict[str, Any]:
    route_counts = dict(results.group_by("route").len().iter_rows()) if not results.is_empty() else {}
    return {
        "schema_version": SHARDED_RUN_VERSION,
        "status": "complete" if failures.is_empty() else "complete_with_operational_failures",
        "started_at": started_at,
        "ended_at": _now(),
        "scientific_scope": "YOLOE visual screening evidence only; not taxonomic validation.",
        "snapshot": {
            "records": prepared.population,
            "sample_seed": config.sample_seed,
            "source_snapshot_fingerprint": prepared.source_snapshot_fingerprint,
            "manifest": str(prepared.manifest_path),
        },
        "runtime": {
            "persistent_models": config.shard_count,
            "device": config.device,
            "checkpoint": config.checkpoint,
            "imgsz": config.imgsz,
            "batch_size": config.yoloe_batch_size,
            "download_workers_per_model": config.download_workers_per_shard,
            "total_download_workers": config.download_workers_per_shard * config.shard_count,
            "prompt_classes": list(config.prompt_classes),
            "prompt_set_fingerprint": yoloe26_prompt_set_fingerprint(config.prompt_classes),
        },
        "counts": {
            "classified": results.height,
            "terminal_operational_failures": failures.height,
            "routes": route_counts,
        },
        "shards": shard_reports,
    }


def _markdown_summary(report: dict[str, Any]) -> str:
    return (
        "# Sharded YOLOE Flickr screening\n\n"
        f"Status: {report['status']}  \n"
        f"Snapshot records: {report['snapshot']['records']}  \n"
        f"Classified: {report['counts']['classified']}  \n"
        f"Terminal operational failures: {report['counts']['terminal_operational_failures']}  \n"
        f"Persistent MPS models: {report['runtime']['persistent_models']}  \n"
        f"Prompt classes: `{json.dumps(report['runtime']['prompt_classes'])}`\n"
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()
