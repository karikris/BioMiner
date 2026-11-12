"""Immutable float32 embedding parts, safely reused across successive cohorts."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import polars as pl

from biominer.embeddinggemma.common import (
    atomic_json,
    atomic_parquet,
    file_sha256,
    now,
    producer_sha,
    run_lock,
)
from biominer.embeddinggemma.runtime import normalize_vectors

SCHEMA = {"image_hash": pl.String, "embedding": pl.List(pl.Float32)}
LOGGER = logging.getLogger(__name__)


@contextmanager
def _failure_receipt(manifest_path: Path, manifest: dict):
    try:
        yield
    except BaseException as exc:
        manifest.update(
            status="failed", error_type=type(exc).__name__, updated_at=now()
        )
        atomic_json(manifest_path, manifest)
        raise


def read_embeddings(directory: Path) -> pl.DataFrame:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return pl.DataFrame(schema=SCHEMA)
    manifest = json.loads(manifest_path.read_text())
    frames = []
    for part in manifest["parts"]:
        path = directory / part["path"]
        if file_sha256(path) != part["sha256"]:
            raise ValueError("Embedding part checksum mismatch")
        frame = pl.read_parquet(path)
        if frame.height != part["rows"]:
            raise ValueError("Embedding part count mismatch")
        frames.append(frame)
    frame = pl.concat(frames) if frames else pl.DataFrame(schema=SCHEMA)
    if frame["image_hash"].n_unique() != frame.height:
        raise ValueError("Duplicate content in embedding checkpoint")
    if frame.height:
        normalize_vectors(frame["embedding"].to_list())
    return frame


def encode_media(
    media: pl.DataFrame, output_dir: Path, *, encoder, budget: int, batch_size: int = 4
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output_dir = Path(output_dir)
    fingerprint = encoder.config.fingerprint(budget)
    with run_lock(output_dir):
        manifest_path = output_dir / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists()
            else {
                "schema_version": "embeddinggemma-embeddings-v1",
                "fingerprint": fingerprint,
                "producer_sha": producer_sha(),
                "created_at": now(),
                "parts": [],
                "budget": budget,
                "dimension": 768,
                "dtype": "float32",
                "evidence_maturity": "experimental_model_evidence",
                "completed_batches": [],
            }
        )
        if manifest["fingerprint"] != fingerprint:
            raise ValueError(
                "Embedding checkpoint fingerprint changed; use a new output directory"
            )
        done = set(read_embeddings(output_dir)["image_hash"].to_list())
        candidates = (
            media.filter(pl.col("status") == "ok")
            .unique("image_hash")
            .sort("image_hash")
        )
        pending = [
            row
            for row in candidates.iter_rows(named=True)
            if row["image_hash"] not in done
        ]
        manifest.update(
            status="running", updated_at=now(), expected_cached_images=candidates.height
        )
        atomic_json(manifest_path, manifest)
        started = time.monotonic()
        with _failure_receipt(manifest_path, manifest):
            _encode_pending(
                pending,
                encoder,
                budget,
                batch_size,
                output_dir,
                manifest,
                manifest_path,
                done,
                started,
            )
        completed = manifest.get("encoded_total", len(done)) - len(done)
        manifest.update(status="complete", updated_at=now())
        atomic_json(manifest_path, manifest)
        return {
            "images_encoded": completed,
            "cache_hits": len(done & set(candidates["image_hash"])),
            "seconds": time.monotonic() - started,
            "fingerprint": fingerprint,
            "metrics": encoder.metrics(),
            "encoded_total": len(done) + completed,
        }


def _encode_pending(
    pending,
    encoder,
    budget,
    batch_size,
    output_dir,
    manifest,
    manifest_path,
    done,
    started,
):
    completed = 0
    index = 0
    while index < len(pending):
        batch = pending[index : index + batch_size]
        manifest["active_batch_hashes"] = [row["image_hash"] for row in batch]
        for row in batch:
            if file_sha256(row["local_path"]) != row["image_hash"]:
                raise ValueError("Cached media checksum changed")
        before = time.monotonic()
        try:
            vectors = normalize_vectors(
                encoder.encode_images(
                    [Path(row["local_path"]) for row in batch], budget
                )
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size == 1:
                manifest.update(
                    status="failed", error_type=type(exc).__name__, updated_at=now()
                )
                atomic_json(manifest_path, manifest)
                raise
            batch_size = max(1, batch_size // 2)
            import torch

            if encoder.config.device == "mps":
                torch.mps.empty_cache()
            continue
        if len(vectors) != len(batch):
            raise ValueError("Encoder returned wrong batch count")
        part = output_dir / f"part-{len(manifest['parts']):06d}.parquet"
        # Unlisted parts can be replaced after a crash before manifest commit.
        atomic_parquet(
            part,
            pl.DataFrame(
                {
                    "image_hash": [r["image_hash"] for r in batch],
                    "embedding": vectors.tolist(),
                },
                schema=SCHEMA,
            ),
        )
        elapsed = time.monotonic() - before
        manifest["parts"].append(
            {"path": part.name, "sha256": file_sha256(part), "rows": len(batch)}
        )
        manifest["completed_batches"].append({"rows": len(batch), "seconds": elapsed})
        index += len(batch)
        completed += len(batch)
        manifest.pop("active_batch_hashes", None)
        cumulative = sum(row["seconds"] for row in manifest["completed_batches"])
        cumulative_rows = sum(row["rows"] for row in manifest["completed_batches"])
        manifest.update(
            updated_at=now(),
            encoded_total=len(done) + completed,
            metrics={
                **encoder.metrics(),
                "inference_seconds": cumulative,
                "images_per_second": cumulative_rows / max(cumulative, 1e-9),
                "raw_image_embedding_bytes": (len(done) + completed) * 768 * 4,
            },
        )
        atomic_json(manifest_path, manifest)
        if completed == len(batch) or completed % 100 < len(batch):
            LOGGER.warning(
                "budget=%s encoded=%s/%s %.2f images/s",
                budget,
                completed,
                len(pending),
                completed / max(time.monotonic() - started, 1e-9),
            )
