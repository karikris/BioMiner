"""Deterministic Flickr query-hit projection for geographic workload builds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import polars as pl


FLICKR_QUERY_HITS_FILE = "flickr_query_hits.parquet"
FLICKR_WORKLOAD_INPUT_SCHEMA_VERSION = "flickr-workload-input-v1.0.0"

_REQUIRED_COLUMNS = frozenset(
    {
        "fetched_at",
        "flickr_photo_id",
        "query_field",
        "query_hash",
        "query_term",
        "query_term_confidence",
        "query_term_type",
        "raw_photo_json",
    }
)
_PROJECTED_COLUMNS = (
    "source",
    "flickr_photo_id",
    "fetched_at",
    "query_hash",
    "query_field",
    "query_term",
    "query_term_confidence",
    "query_term_type",
    "raw_photo_json",
    "latitude",
    "longitude",
    "accuracy",
    "inferred_country_from_bbox",
)


@dataclass(frozen=True, slots=True)
class FlickrWorkloadInput:
    """Canonical photos plus every distinct query-hit provenance row."""

    canonical_photos: pl.DataFrame
    query_hits: pl.DataFrame
    input_row_count: int
    canonical_photo_count: int
    query_hit_count: int


def read_flickr_workload_input(
    path: str | Path,
    *,
    input_format: str = "auto",
) -> FlickrWorkloadInput:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Flickr candidate metadata does not exist: {source}")
    resolved_format = _input_format(source, input_format)
    scan = (
        pl.scan_ndjson(source)
        if resolved_format == "ndjson"
        else pl.scan_parquet(source)
    )
    available = set(scan.collect_schema().names())
    selected = [name for name in _PROJECTED_COLUMNS if name in available]
    return canonicalize_flickr_workload_hits(scan.select(selected).collect())


def canonicalize_flickr_workload_hits(hits: pl.DataFrame) -> FlickrWorkloadInput:
    if not isinstance(hits, pl.DataFrame):
        raise TypeError("hits must be a Polars DataFrame")
    missing = sorted(_REQUIRED_COLUMNS - set(hits.columns))
    if missing:
        raise ValueError(f"Flickr workload input is missing columns: {missing}")
    if hits.is_empty():
        raise ValueError("Flickr workload input must not be empty")

    normalized = hits.with_columns(
        _source_expression(hits),
        *(
            pl.col(name).cast(pl.String, strict=False).fill_null("").alias(name)
            for name in (
                "flickr_photo_id",
                "fetched_at",
                "query_hash",
                "query_field",
                "query_term",
                "query_term_confidence",
                "query_term_type",
                "raw_photo_json",
            )
        ),
    )
    for field in ("flickr_photo_id", "fetched_at", "query_hash", "raw_photo_json"):
        if normalized.filter(pl.col(field).str.strip_chars() == "").height:
            raise ValueError(f"Flickr workload input contains blank {field}")
    if normalized.filter(pl.col("source") != "flickr").height:
        raise ValueError("Flickr workload input contains a non-Flickr source")
    normalized = normalized.with_columns(
        pl.col("fetched_at")
        .str.to_datetime(format="%+", strict=False, time_zone="UTC")
        .alias("_canonical_fetched_at")
    )
    if normalized.filter(pl.col("_canonical_fetched_at").is_null()).height:
        raise ValueError("Flickr workload input contains invalid fetched_at timestamps")
    duplicate_hit_count = normalized.height - normalized.unique(
        subset=("source", "flickr_photo_id", "query_hash")
    ).height
    if duplicate_hit_count:
        raise ValueError(
            "Flickr workload input contains duplicate source/photo/query hits: "
            f"{duplicate_hit_count}"
        )

    ordered = normalized.sort(
        ("source", "flickr_photo_id", "_canonical_fetched_at", "query_hash"),
        descending=(False, False, True, False),
    )
    canonical = (
        ordered.unique(
            subset=("source", "flickr_photo_id"),
            keep="first",
            maintain_order=True,
        )
        .with_columns(
            pl.col("raw_photo_json")
            .map_elements(_source_record_sha256, return_dtype=pl.String)
            .alias("source_record_hash")
        )
        .drop("_canonical_fetched_at")
        .sort(("source", "flickr_photo_id"))
    )
    query_hits = (
        normalized.select(
            "source",
            "flickr_photo_id",
            "query_hash",
            pl.concat_str(
                (
                    pl.col("query_term_type").replace("", "unknown"),
                    pl.col("query_term_confidence").replace("", "unknown"),
                    pl.col("query_field").replace("", "unknown"),
                ),
                separator=":",
            ).alias("query_tier"),
            pl.col("query_term").alias("search_term"),
        )
        .sort(("source", "flickr_photo_id", "query_hash"))
    )
    return FlickrWorkloadInput(
        canonical_photos=canonical,
        query_hits=query_hits,
        input_row_count=normalized.height,
        canonical_photo_count=canonical.height,
        query_hit_count=query_hits.height,
    )


def _source_expression(hits: pl.DataFrame) -> pl.Expr:
    if "source" not in hits.columns:
        return pl.lit("flickr").alias("source")
    return (
        pl.col("source")
        .cast(pl.String, strict=False)
        .fill_null("flickr")
        .str.strip_chars()
        .str.to_lowercase()
        .replace("", "flickr")
        .alias("source")
    )


def _source_record_sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _input_format(path: Path, value: str) -> str:
    normalized = str(value or "auto").strip().casefold()
    if normalized == "auto":
        normalized = "parquet" if path.suffix.casefold() == ".parquet" else "ndjson"
    if normalized not in {"ndjson", "parquet"}:
        raise ValueError("input_format must be auto, ndjson, or parquet")
    return normalized


__all__ = [
    "FLICKR_QUERY_HITS_FILE",
    "FLICKR_WORKLOAD_INPUT_SCHEMA_VERSION",
    "FlickrWorkloadInput",
    "canonicalize_flickr_workload_hits",
    "read_flickr_workload_input",
]
