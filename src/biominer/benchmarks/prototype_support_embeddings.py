from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from array import array
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from math import isfinite, sqrt
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image
import polars as pl

from biominer.bioclip.bioclip import PersistentBioClipScorer
from biominer.bioclip.bioclip_worker import decoded_rgb_image_content_hash
from biominer.bioclip.model_registry import BioClipRuntime, ModelConfig
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.prototype_freeze import (
    PROTOTYPE_READINESS_SCHEMA_VERSION,
    PROTOTYPE_SUPPORT_SCHEMA_VERSION,
    prototype_support_schema,
)
from biominer.vision.full_frame_attention import TargetPreprocessingContract


PROTOTYPE_REFERENCE_EMBEDDINGS_FILE = "prototype_reference_embeddings.parquet"
PROTOTYPE_REFERENCE_PROTOTYPES_FILE = "prototype_reference_prototypes.parquet"
PROTOTYPE_VISUAL_NEIGHBOUR_SPECIES_FILE = "prototype_visual_neighbour_species.parquet"
PROTOTYPE_EMBEDDING_FAILURES_FILE = "prototype_reference_embedding_failures.parquet"
PROTOTYPE_EMBEDDING_REPORT_FILE = "prototype_reference_embedding_report.json"
PROTOTYPE_EMBEDDING_SUMMARY_FILE = "prototype_reference_embedding_summary.md"

PROTOTYPE_REFERENCE_EMBEDDINGS_SCHEMA_VERSION = "prototype-reference-embeddings-v1.0.0"
PROTOTYPE_REFERENCE_PROTOTYPES_SCHEMA_VERSION = "prototype-reference-prototypes-v1.0.0"
PROTOTYPE_VISUAL_NEIGHBOUR_SCHEMA_VERSION = "prototype-visual-neighbour-species-v1.0.0"
PROTOTYPE_EMBEDDING_FAILURES_SCHEMA_VERSION = (
    "prototype-reference-embedding-failures-v1.0.0"
)
PROTOTYPE_EMBEDDING_REPORT_SCHEMA_VERSION = (
    "prototype-reference-embedding-report-v1.0.0"
)

_ALLOWED_READINESS = frozenset({"prototype_ready", "prototype_ready_with_shortfalls"})
_ALLOWED_VERIFICATION = frozenset(
    {"human_verified", "provider_high_trust", "provider_supported"}
)
_SHA256_PREFIX = "sha256:"
_UNIT_NORM_TOLERANCE = 1e-5


@dataclass(frozen=True, slots=True)
class PrototypeSupportEmbeddingConfig:
    support_manifest: Path
    support_manifest_sha256: str
    readiness: Path
    readiness_sha256: str
    output_dir: Path
    runtime_python: Path
    hf_cache_dir: Path
    model_name: str
    model_revision: str
    open_clip_version: str
    device: str = "mps"
    batch_size: int = 16
    preprocess_workers: int = 1
    graph_top_k: int = 5
    graph_minimum_similarity: float = -1.0
    resume: bool = True
    overwrite: bool = False
    skip_records: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for field in (
            "support_manifest",
            "readiness",
            "output_dir",
            "runtime_python",
            "hf_cache_dir",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))
        _require_sha256(self.support_manifest_sha256, field="support_manifest_sha256")
        _require_sha256(self.readiness_sha256, field="readiness_sha256")
        for field in ("model_name", "model_revision", "open_clip_version"):
            _required_text(getattr(self, field), field=field)
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device must be auto, cuda, mps, or cpu")
        for field in ("batch_size", "preprocess_workers", "graph_top_k"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        similarity = float(self.graph_minimum_similarity)
        if not isfinite(similarity) or not -1.0 <= similarity <= 1.0:
            raise ValueError("graph_minimum_similarity must be finite and in [-1, 1]")
        object.__setattr__(self, "graph_minimum_similarity", similarity)
        for field in ("resume", "overwrite"):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be a boolean")
        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")
        normalized_skips: list[tuple[str, str]] = []
        seen: set[str] = set()
        for media_id, reason in self.skip_records:
            media = _required_text(media_id, field="skip record media ID")
            if media in seen:
                raise ValueError(f"duplicate skip record: {media}")
            seen.add(media)
            normalized_skips.append(
                (media, _required_text(reason, field="skip record reason"))
            )
        object.__setattr__(self, "skip_records", tuple(sorted(normalized_skips)))

    @classmethod
    def read_json(cls, path: str | Path) -> PrototypeSupportEmbeddingConfig:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("prototype embedding config must be an object")
        allowed = {
            "schema_version",
            "support_manifest",
            "support_manifest_sha256",
            "readiness",
            "readiness_sha256",
            "output_dir",
            "runtime_python",
            "hf_cache_dir",
            "model_name",
            "model_revision",
            "open_clip_version",
            "device",
            "batch_size",
            "preprocess_workers",
            "graph_top_k",
            "graph_minimum_similarity",
            "resume",
            "overwrite",
            "skip_records",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                f"unknown prototype embedding config fields: {sorted(unknown)}"
            )
        if payload.get("schema_version") != "prototype-support-embedding-job-v1.0.0":
            raise ValueError("unsupported prototype embedding config schema")
        skips = payload.get("skip_records", [])
        if not isinstance(skips, list):
            raise TypeError("skip_records must be an array")
        skip_records = []
        for row in skips:
            if not isinstance(row, Mapping) or set(row) != {
                "reference_media_id",
                "reason",
            }:
                raise ValueError(
                    "each skip record requires reference_media_id and reason"
                )
            skip_records.append((str(row["reference_media_id"]), str(row["reason"])))
        values = dict(payload)
        values.pop("schema_version", None)
        values["skip_records"] = tuple(skip_records)
        return cls(**values)  # type: ignore[arg-type]

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": "prototype-support-embedding-job-v1.0.0",
                "support_manifest_sha256": self.support_manifest_sha256,
                "readiness_sha256": self.readiness_sha256,
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "open_clip_version": self.open_clip_version,
                "device": self.device,
                "batch_size": self.batch_size,
                "preprocess_workers": self.preprocess_workers,
                "graph_top_k": self.graph_top_k,
                "graph_minimum_similarity": self.graph_minimum_similarity,
                "resume": self.resume,
                "overwrite": self.overwrite,
                "skip_records": self.skip_records,
            }
        )


@dataclass(frozen=True, slots=True)
class PrototypeSupportEmbeddingResult:
    report: dict[str, object]
    embeddings_path: Path
    prototypes_path: Path
    visual_neighbours_path: Path | None
    failures_path: Path | None
    report_path: Path
    summary_path: Path


def prototype_reference_embeddings_schema(
    embedding_dimension: int,
) -> dict[str, pl.DataType]:
    dimension = _positive_int(embedding_dimension, field="embedding_dimension")
    return {
        "schema_version": pl.String,
        "reference_bank_version": pl.String,
        "support_manifest_fingerprint": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "support_row_fingerprint": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "source": pl.String,
        "trust_level": pl.String,
        "verification_status": pl.String,
        "human_verified": pl.Boolean,
        "geographic_layer": pl.String,
        "geo_cluster_id": pl.String,
        "route": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "reference_group": pl.String,
        "licence": pl.String,
        "licence_policy_status": pl.String,
        "attribution": pl.String,
        "dataset_split": pl.String,
        "leakage_component_id": pl.String,
        "source_image_sha256": pl.String,
        "decoded_image_sha256": pl.String,
        "model_id": pl.String,
        "model_revision": pl.String,
        "model_weights_sha256": pl.String,
        "open_clip_version": pl.String,
        "open_clip_config_sha256": pl.String,
        "preprocessing_version": pl.String,
        "preprocessing_fingerprint": pl.String,
        "embedding_dimension": pl.UInt32,
        "embedding": pl.Array(pl.Float32, dimension),
        "embedding_norm": pl.Float64,
        "prototype_only": pl.Boolean,
        "embedding_fingerprint": pl.String,
    }


def prototype_reference_prototypes_schema(
    embedding_dimension: int,
) -> dict[str, pl.DataType]:
    dimension = _positive_int(embedding_dimension, field="embedding_dimension")
    return {
        "schema_version": pl.String,
        "prototype_id": pl.String,
        "reference_bank_version": pl.String,
        "support_manifest_fingerprint": pl.String,
        "reference_embeddings_fingerprint": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "route": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "scope_type": pl.String,
        "geo_cluster_id": pl.String,
        "prototype_method": pl.String,
        "member_reference_media_ids": pl.List(pl.String),
        "member_observation_ids": pl.List(pl.String),
        "reference_count": pl.UInt32,
        "independent_observation_count": pl.UInt32,
        "embedding_dimension": pl.UInt32,
        "embedding": pl.Array(pl.Float32, dimension),
        "embedding_norm": pl.Float64,
        "model_fingerprint": pl.String,
        "prototype_only": pl.Boolean,
        "prototype_fingerprint": pl.String,
    }


def prototype_visual_neighbour_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "edge_id": pl.String,
        "graph_fingerprint": pl.String,
        "reference_prototypes_fingerprint": pl.String,
        "subject_accepted_taxon_key": pl.String,
        "subject_scientific_name": pl.String,
        "neighbour_accepted_taxon_key": pl.String,
        "neighbour_scientific_name": pl.String,
        "route": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "similarity": pl.Float64,
        "neighbour_rank": pl.UInt32,
        "top_k": pl.UInt32,
        "minimum_similarity": pl.Float64,
        "prototype_only": pl.Boolean,
        "edge_fingerprint": pl.String,
    }


def prototype_embedding_failures_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_bank_version": pl.String,
        "support_manifest_fingerprint": pl.String,
        "reference_media_id": pl.String,
        "support_row_fingerprint": pl.String,
        "dataset_split": pl.String,
        "route": pl.String,
        "failure_stage": pl.String,
        "retryable": pl.Boolean,
        "error_type": pl.String,
        "error_message": pl.String,
        "failure_fingerprint": pl.String,
    }


def run_prototype_support_embedding_job(
    config: PrototypeSupportEmbeddingConfig,
    *,
    scorer: object | None = None,
) -> PrototypeSupportEmbeddingResult:
    started_at = datetime.now(UTC)
    support, readiness = _load_and_validate_inputs(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = config.output_dir / PROTOTYPE_REFERENCE_EMBEDDINGS_FILE
    prototypes_path = config.output_dir / PROTOTYPE_REFERENCE_PROTOTYPES_FILE
    neighbours_path = config.output_dir / PROTOTYPE_VISUAL_NEIGHBOUR_SPECIES_FILE
    failures_path = config.output_dir / PROTOTYPE_EMBEDDING_FAILURES_FILE
    report_path = config.output_dir / PROTOTYPE_EMBEDDING_REPORT_FILE
    summary_path = config.output_dir / PROTOTYPE_EMBEDDING_SUMMARY_FILE
    own_scorer = scorer is None
    effective_scorer = scorer or PersistentBioClipScorer(
        runtime=_bioclip_runtime(config),
        hf_cache_dir=str(config.hf_cache_dir),
        device=config.device,
        image_resize_mode="longest",
        preprocess_workers=config.preprocess_workers,
    )
    failures: list[dict[str, object]] = []
    rows, resumed_ids = _resumed_embedding_rows(
        embeddings_path,
        config=config,
        readiness=readiness,
    )
    previous_report = _read_previous_report(report_path)
    skip_by_id = dict(config.skip_records)
    known_ids = set(support["reference_media_id"].to_list())
    unknown_skips = set(skip_by_id) - known_ids
    if unknown_skips:
        raise ValueError(
            f"skip records are absent from frozen support: {sorted(unknown_skips)}"
        )
    pending: list[dict[str, object]] = []
    for row in support.iter_rows(named=True):
        media_id = str(row["reference_media_id"])
        if media_id in resumed_ids:
            continue
        if media_id in skip_by_id:
            failures.append(
                _failure_row(
                    row,
                    readiness=readiness,
                    failure_stage="operator_skip",
                    error_type="operator_skipped",
                    error_message=skip_by_id[media_id],
                )
            )
        else:
            pending.append(row)
    try:
        if pending:
            _attest_scorer(effective_scorer, config)
        for offset in range(0, len(pending), config.batch_size):
            batch = pending[offset : offset + config.batch_size]
            batch_rows, batch_failures = _embed_batch_resilient(
                batch,
                scorer=effective_scorer,
                readiness=readiness,
            )
            rows.extend(batch_rows)
            failures.extend(batch_failures)
            if rows:
                checkpoint = _embedding_frame(rows)
                _atomic_write_parquet(
                    checkpoint,
                    embeddings_path,
                )
            if failures:
                _atomic_write_parquet(
                    _failures_frame(failures),
                    failures_path,
                )
    finally:
        if own_scorer:
            close = getattr(effective_scorer, "close", None)
            if callable(close):
                close()
    if not rows:
        raise RuntimeError("prototype embedding job produced no successful embeddings")
    embeddings = _embedding_frame(rows)
    validate_prototype_reference_embeddings(embeddings)
    prototypes = build_prototype_reference_prototypes(embeddings)
    neighbours = build_prototype_visual_neighbours(
        prototypes,
        top_k=config.graph_top_k,
        minimum_similarity=config.graph_minimum_similarity,
    )
    _atomic_write_parquet(embeddings, embeddings_path)
    _atomic_write_parquet(prototypes, prototypes_path)
    if neighbours.is_empty():
        neighbours_path = None
    else:
        _atomic_write_parquet(neighbours, neighbours_path)
    if not failures:
        failures_path.unlink(missing_ok=True)
        failures_path = None
    completed_ids = set(embeddings["reference_media_id"].to_list())
    failed_ids = {str(row["reference_media_id"]) for row in failures}
    if completed_ids & failed_ids or completed_ids | failed_ids != known_ids:
        raise RuntimeError("prototype embedding completion partition is incomplete")
    ended_at = datetime.now(UTC)
    report: dict[str, object] = {
        "schema_version": PROTOTYPE_EMBEDDING_REPORT_SCHEMA_VERSION,
        "status": "complete_with_retryable_failures" if failures else "complete",
        "prototype_only": True,
        "experimental_screening_evidence_only": True,
        "configuration_fingerprint": config.fingerprint,
        "reference_bank_version": readiness["reference_bank_version"],
        "support_manifest_fingerprint": readiness["support_manifest_fingerprint"],
        "support_manifest_sha256": config.support_manifest_sha256,
        "readiness_sha256": config.readiness_sha256,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "model": _model_report(
            effective_scorer,
            embeddings=embeddings,
            executed=bool(pending),
            previous_report=previous_report,
        ),
        "counts": {
            "frozen_support": support.height,
            "embedded": embeddings.height,
            "resumed_embeddings": len(resumed_ids),
            "retryable_failures": len(failures),
            "operator_skips": sum(
                row["error_type"] == "operator_skipped" for row in failures
            ),
            "prototypes": prototypes.height,
            "visual_neighbour_edges": neighbours.height,
        },
        "route_counts": _counts(embeddings, "route"),
        "split_counts": _counts(embeddings, "dataset_split"),
        "artifacts": {
            "prototype_reference_embeddings": _artifact_record(embeddings_path),
            "prototype_reference_prototypes": _artifact_record(prototypes_path),
            "prototype_visual_neighbour_species": (
                _artifact_record(neighbours_path) if neighbours_path else None
            ),
            "prototype_reference_embedding_failures": (
                _artifact_record(failures_path) if failures_path else None
            ),
        },
        "fingerprints": {
            "embeddings": _frame_fingerprint(embeddings, "embedding_fingerprint"),
            "prototypes": _frame_fingerprint(prototypes, "prototype_fingerprint"),
            "visual_neighbours": (
                str(neighbours["graph_fingerprint"][0])
                if not neighbours.is_empty()
                else None
            ),
        },
        "semantics": {
            "provider_supported_is_human_verified": False,
            "model_output_is_taxonomic_validation": False,
            "operational_failures_are_biological_negatives": False,
            "adult_larval_and_specimen_embeddings_are_mixed": False,
            "prototypes_consume_only_support_train": True,
        },
    }
    report["report_fingerprint"] = canonical_semantic_fingerprint(
        {key: value for key, value in report.items() if key != "report_fingerprint"}
    )
    _atomic_write_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_write_text(summary_path, _summary(report))
    return PrototypeSupportEmbeddingResult(
        report=report,
        embeddings_path=embeddings_path,
        prototypes_path=prototypes_path,
        visual_neighbours_path=neighbours_path,
        failures_path=failures_path,
        report_path=report_path,
        summary_path=summary_path,
    )


def validate_prototype_reference_embeddings(frame: pl.DataFrame) -> None:
    if frame.is_empty():
        raise ValueError("prototype reference embeddings must not be empty")
    dimensions = frame["embedding_dimension"].unique().to_list()
    if len(dimensions) != 1:
        raise ValueError("prototype reference embeddings have mixed dimensions")
    dimension = _positive_int(dimensions[0], field="embedding_dimension")
    if dict(frame.schema) != prototype_reference_embeddings_schema(dimension):
        raise ValueError("prototype reference embeddings physical schema mismatch")
    if not frame.equals(_sort_embeddings(frame)):
        raise ValueError(
            "prototype reference embeddings are not deterministically sorted"
        )
    if frame["reference_media_id"].n_unique() != frame.height:
        raise ValueError("prototype reference embeddings contain duplicate media")
    if frame.filter(~pl.col("prototype_only")).height:
        raise ValueError("prototype reference embeddings contain non-prototype rows")
    if frame.filter(
        pl.col("human_verified") & (pl.col("verification_status") != "human_verified")
    ).height:
        raise ValueError("prototype embeddings overstate human verification")
    route_dimensions: dict[str, set[tuple[str, str]]] = {
        "adult_field": {("adult", "live_field")},
        "larval": {("larva", "live_field")},
        "pinned_specimen": {
            ("adult", "pinned_specimen"),
            ("unknown", "pinned_specimen"),
        },
    }
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != PROTOTYPE_REFERENCE_EMBEDDINGS_SCHEMA_VERSION:
            raise ValueError("unsupported prototype embedding schema")
        if row["verification_status"] not in _ALLOWED_VERIFICATION:
            raise ValueError(
                "prototype embedding uses ineligible verification evidence"
            )
        expected = route_dimensions.get(str(row["route"]))
        if (
            expected is None
            or (row["life_stage"], row["visual_domain"]) not in expected
        ):
            raise ValueError("prototype embedding route dimensions mismatch")
        vector = tuple(float(value) for value in row["embedding"])
        norm = sqrt(sum(value * value for value in vector))
        if len(vector) != dimension or any(not isfinite(value) for value in vector):
            raise ValueError("prototype embedding vector is invalid")
        if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
            raise ValueError("prototype embedding vector is not unit normalized")
        if abs(norm - float(row["embedding_norm"])) > 1e-12:
            raise ValueError("prototype embedding stored norm is invalid")
        if row["embedding_fingerprint"] != _row_fingerprint(
            row, "embedding_fingerprint"
        ):
            raise ValueError("prototype embedding fingerprint mismatch")


def build_prototype_reference_prototypes(embeddings: pl.DataFrame) -> pl.DataFrame:
    validate_prototype_reference_embeddings(embeddings)
    support = embeddings.filter(pl.col("dataset_split") == "support_train")
    if support.is_empty():
        raise ValueError("prototype construction requires support_train embeddings")
    dimension = int(support["embedding_dimension"][0])
    embeddings_fingerprint = _frame_fingerprint(support, "embedding_fingerprint")
    rows: list[dict[str, object]] = []
    group_fields = [
        "accepted_taxon_key",
        "scientific_name",
        "route",
        "life_stage",
        "visual_domain",
    ]
    for keys, group in support.group_by(group_fields, maintain_order=False):
        values = dict(zip(group_fields, keys, strict=True))
        rows.append(
            _prototype_row(
                group,
                values=values,
                scope_type="global",
                geo_cluster_id="all",
                embeddings_fingerprint=embeddings_fingerprint,
            )
        )
        for (geo_cluster_id,), local in group.group_by(
            "geo_cluster_id", maintain_order=False
        ):
            rows.append(
                _prototype_row(
                    local,
                    values=values,
                    scope_type="regional",
                    geo_cluster_id=str(geo_cluster_id),
                    embeddings_fingerprint=embeddings_fingerprint,
                )
            )
    result = pl.DataFrame(
        rows,
        schema=prototype_reference_prototypes_schema(dimension),
        orient="row",
        strict=True,
    ).sort(
        [
            "route",
            "accepted_taxon_key",
            "scope_type",
            "geo_cluster_id",
            "prototype_id",
        ]
    )
    _validate_prototypes(result)
    return result


def build_prototype_visual_neighbours(
    prototypes: pl.DataFrame,
    *,
    top_k: int = 5,
    minimum_similarity: float = -1.0,
) -> pl.DataFrame:
    _validate_prototypes(prototypes)
    _positive_int(top_k, field="top_k")
    if not isfinite(minimum_similarity) or not -1.0 <= minimum_similarity <= 1.0:
        raise ValueError("minimum_similarity must be finite and in [-1, 1]")
    global_rows = prototypes.filter(
        (pl.col("scope_type") == "global")
        & (pl.col("geo_cluster_id") == "all")
        & (pl.col("prototype_method") == "normalized_observation_mean")
    )
    prototypes_fingerprint = _frame_fingerprint(prototypes, "prototype_fingerprint")
    rows: list[dict[str, object]] = []
    by_route: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in global_rows.iter_rows(named=True):
        by_route[str(row["route"])].append(row)
    for route, route_rows in sorted(by_route.items()):
        ordered = sorted(route_rows, key=lambda row: str(row["accepted_taxon_key"]))
        for subject in ordered:
            candidates = []
            for neighbour in ordered:
                if neighbour["accepted_taxon_key"] == subject["accepted_taxon_key"]:
                    continue
                similarity = _dot(subject["embedding"], neighbour["embedding"])
                if similarity >= minimum_similarity:
                    candidates.append((similarity, neighbour))
            candidates.sort(
                key=lambda item: (-item[0], str(item[1]["accepted_taxon_key"]))
            )
            for rank, (similarity, neighbour) in enumerate(candidates[:top_k], start=1):
                base = {
                    "schema_version": PROTOTYPE_VISUAL_NEIGHBOUR_SCHEMA_VERSION,
                    "reference_prototypes_fingerprint": prototypes_fingerprint,
                    "subject_accepted_taxon_key": subject["accepted_taxon_key"],
                    "subject_scientific_name": subject["scientific_name"],
                    "neighbour_accepted_taxon_key": neighbour["accepted_taxon_key"],
                    "neighbour_scientific_name": neighbour["scientific_name"],
                    "route": route,
                    "life_stage": subject["life_stage"],
                    "visual_domain": subject["visual_domain"],
                    "similarity": similarity,
                    "neighbour_rank": rank,
                    "top_k": top_k,
                    "minimum_similarity": minimum_similarity,
                    "prototype_only": True,
                }
                edge_fingerprint = canonical_semantic_fingerprint(base)
                rows.append(
                    {
                        **base,
                        "edge_id": "prototype-visual-edge:" + edge_fingerprint[7:],
                        "edge_fingerprint": edge_fingerprint,
                    }
                )
    if not rows:
        return pl.DataFrame(schema=prototype_visual_neighbour_schema())
    edge_fingerprints = sorted(str(row["edge_fingerprint"]) for row in rows)
    graph_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": PROTOTYPE_VISUAL_NEIGHBOUR_SCHEMA_VERSION,
            "reference_prototypes_fingerprint": prototypes_fingerprint,
            "top_k": top_k,
            "minimum_similarity": minimum_similarity,
            "edge_fingerprints": edge_fingerprints,
        }
    )
    for row in rows:
        row["graph_fingerprint"] = graph_fingerprint
    return pl.DataFrame(
        rows,
        schema=prototype_visual_neighbour_schema(),
        orient="row",
        strict=True,
    ).sort(
        [
            "route",
            "subject_accepted_taxon_key",
            "neighbour_rank",
            "neighbour_accepted_taxon_key",
        ]
    )


def _load_and_validate_inputs(
    config: PrototypeSupportEmbeddingConfig,
) -> tuple[pl.DataFrame, dict[str, object]]:
    _verify_file(config.support_manifest, config.support_manifest_sha256)
    _verify_file(config.readiness, config.readiness_sha256)
    support = pl.read_parquet(config.support_manifest)
    if dict(support.schema) != prototype_support_schema():
        raise ValueError("prototype support manifest physical schema mismatch")
    if support.is_empty() or support["reference_media_id"].n_unique() != support.height:
        raise ValueError("prototype support manifest is empty or contains duplicates")
    if not support.equals(support.sort("reference_media_id")):
        raise ValueError("prototype support manifest is not deterministically sorted")
    if set(support["schema_version"].to_list()) != {PROTOTYPE_SUPPORT_SCHEMA_VERSION}:
        raise ValueError("unsupported prototype support schema")
    if support.filter(~pl.col("prototype_only")).height:
        raise ValueError("prototype support manifest contains non-prototype rows")
    if support.filter(~pl.col("attribution_complete")).height:
        raise ValueError("prototype support manifest has incomplete attribution")
    if support.filter(
        ~pl.col("verification_status").is_in(_ALLOWED_VERIFICATION)
    ).height:
        raise ValueError(
            "prototype support manifest has ineligible verification evidence"
        )
    readiness = json.loads(config.readiness.read_text(encoding="utf-8"))
    if not isinstance(readiness, dict):
        raise TypeError("prototype readiness must be an object")
    if readiness.get("schema_version") != PROTOTYPE_READINESS_SCHEMA_VERSION:
        raise ValueError("unsupported prototype readiness schema")
    if readiness.get("prototype_readiness_status") not in _ALLOWED_READINESS:
        raise ValueError("prototype readiness does not authorize classification")
    if readiness.get("classification_authorised") is not True:
        raise ValueError("prototype readiness does not authorize classification")
    if readiness.get("bank_status") != "prototype_only":
        raise ValueError("prototype readiness is not prototype-only")
    if readiness.get("human_verification_complete") is not False:
        raise ValueError("prototype readiness overstates human verification")
    actual_fingerprint = canonical_semantic_fingerprint(support.to_dicts())
    if readiness.get("support_manifest_fingerprint") != actual_fingerprint:
        raise ValueError("prototype support semantic fingerprint mismatch")
    return support, readiness


def _resumed_embedding_rows(
    path: Path,
    *,
    config: PrototypeSupportEmbeddingConfig,
    readiness: Mapping[str, object],
) -> tuple[list[dict[str, object]], set[str]]:
    generated_paths = (
        path,
        path.parent / PROTOTYPE_REFERENCE_PROTOTYPES_FILE,
        path.parent / PROTOTYPE_VISUAL_NEIGHBOUR_SPECIES_FILE,
        path.parent / PROTOTYPE_EMBEDDING_FAILURES_FILE,
        path.parent / PROTOTYPE_EMBEDDING_REPORT_FILE,
        path.parent / PROTOTYPE_EMBEDDING_SUMMARY_FILE,
    )
    if config.overwrite:
        for generated in generated_paths:
            generated.unlink(missing_ok=True)
        return [], set()
    if not path.exists():
        return [], set()
    if not config.resume:
        raise FileExistsError(
            "prototype embedding output exists; enable resume or overwrite"
        )
    frame = pl.read_parquet(path)
    validate_prototype_reference_embeddings(frame)
    expected_singletons = {
        "reference_bank_version": readiness["reference_bank_version"],
        "support_manifest_fingerprint": readiness["support_manifest_fingerprint"],
        "model_id": config.model_name,
        "model_revision": config.model_revision,
        "open_clip_version": config.open_clip_version,
    }
    for field, expected in expected_singletons.items():
        observed = frame[field].unique().to_list()
        if observed != [expected]:
            raise ValueError(
                f"prototype embedding resume {field} does not match configuration"
            )
    rows = frame.to_dicts()
    return rows, set(str(value) for value in frame["reference_media_id"].to_list())


def _embed_batch_resilient(
    batch: Sequence[dict[str, object]],
    *,
    scorer: object,
    readiness: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    try:
        return _embed_exact_batch(batch, scorer=scorer, readiness=readiness), []
    except Exception as batch_exc:  # noqa: BLE001 - isolate the failed record.
        if len(batch) == 1:
            return [], [
                _failure_row(
                    batch[0],
                    readiness=readiness,
                    failure_stage="decode_or_embedding",
                    error_type=type(batch_exc).__name__,
                    error_message=str(batch_exc),
                )
            ]
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for row in batch:
        try:
            rows.extend(_embed_exact_batch([row], scorer=scorer, readiness=readiness))
        except Exception as exc:  # noqa: BLE001 - preserve a retryable record.
            failures.append(
                _failure_row(
                    row,
                    readiness=readiness,
                    failure_stage="decode_or_embedding",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
    return rows, failures


def _embed_exact_batch(
    batch: Sequence[dict[str, object]],
    *,
    scorer: object,
    readiness: Mapping[str, object],
) -> list[dict[str, object]]:
    paths: list[Path] = []
    decoded_hashes: list[str] = []
    for row in batch:
        path = Path(str(row["source_object_uri"]))
        _verify_file(path, str(row["source_image_sha256"]))
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.load()
            decoded_hashes.append(decoded_rgb_image_content_hash(rgb))
        paths.append(path)
    embed = getattr(scorer, "embed_image_paths", None)
    if not callable(embed):
        raise TypeError("prototype embedding scorer lacks embed_image_paths")
    raw_vectors = embed(paths)
    if getattr(scorer, "effective_image_resize_mode", None) != "longest":
        raise RuntimeError("BioCLIP did not apply longest-side preprocessing")
    if len(raw_vectors) != len(batch):
        raise RuntimeError("BioCLIP returned the wrong embedding row count")
    scorer_hashes = tuple(getattr(scorer, "last_image_content_hashes", ()))
    if scorer_hashes != tuple(decoded_hashes):
        raise RuntimeError("BioCLIP decoded content hashes do not match inputs")
    rows = []
    for support, decoded_hash, raw_vector in zip(
        batch, decoded_hashes, raw_vectors, strict=True
    ):
        vector, norm = _unit_vector(raw_vector)
        row = {
            "schema_version": PROTOTYPE_REFERENCE_EMBEDDINGS_SCHEMA_VERSION,
            "reference_bank_version": readiness["reference_bank_version"],
            "support_manifest_fingerprint": readiness["support_manifest_fingerprint"],
            "reference_media_id": support["reference_media_id"],
            "reference_observation_id": support["reference_observation_id"],
            "support_row_fingerprint": support["support_row_fingerprint"],
            "accepted_taxon_key": support["accepted_taxon_key"],
            "scientific_name": support["scientific_name"],
            "source": support["source"],
            "trust_level": support["trust_level"],
            "verification_status": support["verification_status"],
            "human_verified": support["human_verified"],
            "geographic_layer": support["geographic_layer"],
            "geo_cluster_id": support["geo_cluster_id"],
            "route": support["route"],
            "life_stage": support["life_stage"],
            "visual_domain": support["visual_domain"],
            "reference_group": support["reference_group"],
            "licence": support["licence"],
            "licence_policy_status": support["licence_policy_status"],
            "attribution": support["attribution"],
            "dataset_split": support["dataset_split"],
            "leakage_component_id": support["leakage_component_id"],
            "source_image_sha256": support["source_image_sha256"],
            "decoded_image_sha256": decoded_hash,
            "model_id": getattr(scorer, "model_id"),
            "model_revision": getattr(scorer, "model_revision"),
            "model_weights_sha256": getattr(scorer, "model_weights_sha256"),
            "open_clip_version": getattr(scorer, "open_clip_version"),
            "open_clip_config_sha256": getattr(scorer, "open_clip_config_sha256"),
            "preprocessing_version": TargetPreprocessingContract().version,
            "preprocessing_fingerprint": getattr(scorer, "preprocessing_fingerprint"),
            "embedding_dimension": len(vector),
            "embedding": vector,
            "embedding_norm": norm,
            "prototype_only": True,
        }
        row["embedding_fingerprint"] = _row_fingerprint(row, "embedding_fingerprint")
        rows.append(row)
    return rows


def _embedding_frame(rows: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    dimensions = {len(row["embedding"]) for row in rows}  # type: ignore[arg-type]
    if len(dimensions) != 1:
        raise ValueError("prototype embedding rows have mixed dimensions")
    dimension = next(iter(dimensions))
    return _sort_embeddings(
        pl.DataFrame(
            rows,
            schema=prototype_reference_embeddings_schema(dimension),
            orient="row",
            strict=True,
        )
    )


def _failures_frame(rows: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=prototype_embedding_failures_schema(),
        orient="row",
        strict=True,
    ).sort("reference_media_id")


def _failure_row(
    support: Mapping[str, object],
    *,
    readiness: Mapping[str, object],
    failure_stage: str,
    error_type: str,
    error_message: str,
) -> dict[str, object]:
    base = {
        "schema_version": PROTOTYPE_EMBEDDING_FAILURES_SCHEMA_VERSION,
        "reference_bank_version": readiness["reference_bank_version"],
        "support_manifest_fingerprint": readiness["support_manifest_fingerprint"],
        "reference_media_id": support["reference_media_id"],
        "support_row_fingerprint": support["support_row_fingerprint"],
        "dataset_split": support["dataset_split"],
        "route": support["route"],
        "failure_stage": failure_stage,
        "retryable": True,
        "error_type": _required_text(error_type, field="error_type"),
        "error_message": _required_text(error_message, field="error_message")[:2000],
    }
    return {**base, "failure_fingerprint": canonical_semantic_fingerprint(base)}


def _prototype_row(
    group: pl.DataFrame,
    *,
    values: Mapping[str, object],
    scope_type: str,
    geo_cluster_id: str,
    embeddings_fingerprint: str,
) -> dict[str, object]:
    observations: dict[str, list[Sequence[float]]] = defaultdict(list)
    media_by_observation: dict[str, list[str]] = defaultdict(list)
    for row in group.iter_rows(named=True):
        observation_id = str(row["reference_observation_id"])
        observations[observation_id].append(row["embedding"])
        media_by_observation[observation_id].append(str(row["reference_media_id"]))
    observation_vectors = [
        _unit_vector(_mean(vectors))[0] for _, vectors in sorted(observations.items())
    ]
    vector, norm = _unit_vector(_mean(observation_vectors))
    member_observations = sorted(observations)
    member_media = sorted(
        media_id
        for media_ids in media_by_observation.values()
        for media_id in media_ids
    )
    model_fingerprint = canonical_semantic_fingerprint(
        {
            "model_id": group["model_id"][0],
            "model_revision": group["model_revision"][0],
            "model_weights_sha256": group["model_weights_sha256"][0],
            "open_clip_config_sha256": group["open_clip_config_sha256"][0],
            "preprocessing_fingerprint": group["preprocessing_fingerprint"][0],
            "embedding_dimension": len(vector),
        }
    )
    base = {
        "schema_version": PROTOTYPE_REFERENCE_PROTOTYPES_SCHEMA_VERSION,
        "reference_bank_version": group["reference_bank_version"][0],
        "support_manifest_fingerprint": group["support_manifest_fingerprint"][0],
        "reference_embeddings_fingerprint": embeddings_fingerprint,
        **values,
        "scope_type": scope_type,
        "geo_cluster_id": geo_cluster_id,
        "prototype_method": "normalized_observation_mean",
        "member_reference_media_ids": member_media,
        "member_observation_ids": member_observations,
        "reference_count": len(member_media),
        "independent_observation_count": len(member_observations),
        "embedding_dimension": len(vector),
        "embedding": vector,
        "embedding_norm": norm,
        "model_fingerprint": model_fingerprint,
        "prototype_only": True,
    }
    fingerprint = _row_fingerprint(base, "prototype_fingerprint")
    return {
        **base,
        "prototype_id": "prototype-reference:" + fingerprint[7:],
        "prototype_fingerprint": fingerprint,
    }


def _validate_prototypes(frame: pl.DataFrame) -> None:
    if frame.is_empty():
        raise ValueError("prototype reference prototypes must not be empty")
    dimensions = frame["embedding_dimension"].unique().to_list()
    if len(dimensions) != 1 or dict(frame.schema) != (
        prototype_reference_prototypes_schema(int(dimensions[0]))
    ):
        raise ValueError("prototype reference prototypes physical schema mismatch")
    if frame["prototype_id"].n_unique() != frame.height:
        raise ValueError("prototype reference prototypes contain duplicate IDs")
    for row in frame.iter_rows(named=True):
        vector = tuple(float(value) for value in row["embedding"])
        if abs(sqrt(sum(value * value for value in vector)) - 1.0) > (
            _UNIT_NORM_TOLERANCE
        ):
            raise ValueError("prototype vector is not unit normalized")
        if row["prototype_fingerprint"] != _row_fingerprint(
            {key: value for key, value in row.items() if key != "prototype_id"},
            "prototype_fingerprint",
        ):
            raise ValueError("prototype fingerprint mismatch")


def _attest_scorer(scorer: object, config: PrototypeSupportEmbeddingConfig) -> None:
    ensure = getattr(scorer, "ensure_model_attestation", None)
    if callable(ensure):
        ensure()
    expected = {
        "model_id": config.model_name,
        "model_revision": config.model_revision,
        "open_clip_version": config.open_clip_version,
    }
    for field, value in expected.items():
        if getattr(scorer, field, None) != value:
            raise ValueError(f"BioCLIP {field} does not match frozen configuration")
    for field in (
        "model_weights_sha256",
        "open_clip_config_sha256",
        "preprocessing_fingerprint",
    ):
        _require_sha256(getattr(scorer, field, None), field=f"BioCLIP {field}")
    if getattr(scorer, "image_resize_mode", None) != "longest":
        raise ValueError("prototype embeddings require longest-side preprocessing")
    effective_resize_mode = getattr(scorer, "effective_image_resize_mode", None)
    if effective_resize_mode not in {None, "longest"}:
        raise ValueError("BioCLIP effective preprocessing is not longest-side")


def _bioclip_runtime(config: PrototypeSupportEmbeddingConfig) -> BioClipRuntime:
    return BioClipRuntime(
        model=ModelConfig(
            model_id="bioclip2_5_huge",
            display_name="BioCLIP 2.5 Huge",
            role="preferred",
            status="use_if_available",
            task="frozen prototype support embedding",
            model_name=config.model_name,
            checkpoint=config.model_revision,
            package_name="open_clip_torch",
            package_version=config.open_clip_version,
            model_hash=f"hf-revision:{config.model_revision}",
        ),
        home=config.runtime_python.parent.parent,
        venv_python=_absolute(config.runtime_python),
        package_version=config.open_clip_version,
        available=True,
    )


def _sort_embeddings(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort(
        [
            "route",
            "accepted_taxon_key",
            "dataset_split",
            "reference_media_id",
        ]
    )


def _unit_vector(values: Sequence[float]) -> tuple[list[float], float]:
    vector = [float(value) for value in values]
    if not vector or any(not isfinite(value) for value in vector):
        raise ValueError("embedding vector must contain finite values")
    raw_norm = sqrt(sum(value * value for value in vector))
    if not isfinite(raw_norm) or raw_norm <= 0:
        raise ValueError("embedding vector must have a nonzero finite norm")
    try:
        stored = [float(value) for value in array("f", (v / raw_norm for v in vector))]
    except OverflowError as exc:
        raise ValueError("embedding vector must fit Float32") from exc
    stored_norm = sqrt(sum(value * value for value in stored))
    if not isfinite(stored_norm) or abs(stored_norm - 1.0) > _UNIT_NORM_TOLERANCE:
        raise ValueError("stored embedding vector is not unit normalized")
    return stored, stored_norm


def _mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot average an empty vector collection")
    dimension = len(vectors[0])
    if dimension == 0 or any(len(vector) != dimension for vector in vectors):
        raise ValueError("cannot average mixed embedding dimensions")
    return [
        sum(float(vector[index]) for vector in vectors) / len(vectors)
        for index in range(dimension)
    ]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("cannot compare mixed embedding dimensions")
    return max(-1.0, min(1.0, sum(float(a) * float(b) for a, b in zip(left, right))))


def _row_fingerprint(row: Mapping[str, object], field: str) -> str:
    return canonical_semantic_fingerprint(
        {key: value for key, value in row.items() if key != field}
    )


def _frame_fingerprint(frame: pl.DataFrame, column: str) -> str:
    return canonical_semantic_fingerprint(sorted(frame[column].to_list()))


def _counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    return dict(
        sorted(
            (str(value), int(count))
            for value, count in frame.group_by(column).len().iter_rows()
        )
    )


def _model_report(
    scorer: object,
    *,
    embeddings: pl.DataFrame,
    executed: bool,
    previous_report: Mapping[str, object] | None,
) -> dict[str, object]:
    persisted_fields = (
        "model_id",
        "model_revision",
        "model_weights_sha256",
        "open_clip_version",
        "open_clip_config_sha256",
        "preprocessing_version",
        "preprocessing_fingerprint",
    )
    report = {field: embeddings[field][0] for field in persisted_fields}
    previous_model = (
        previous_report.get("model") if previous_report is not None else None
    )
    if not isinstance(previous_model, Mapping):
        previous_model = {}
    report.update(
        {
            "effective_image_resize_mode": (
                getattr(scorer, "effective_image_resize_mode", None)
                if executed
                else previous_model.get("effective_image_resize_mode", "longest")
            ),
            "device": (
                getattr(scorer, "device", None)
                if executed
                else previous_model.get("device")
            ),
            "gpu_name": (
                getattr(scorer, "gpu_name", None)
                if executed
                else previous_model.get("gpu_name")
            ),
            "cache_metrics": (
                getattr(scorer, "cache_metrics", None)
                if executed
                else previous_model.get("cache_metrics")
            ),
            "model_execution_performed_this_run": executed,
            "resumed_runtime_attestation": bool(previous_model) and not executed,
        }
    )
    return report


def _read_previous_report(path: Path) -> Mapping[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("schema_version") != PROTOTYPE_EMBEDDING_REPORT_SCHEMA_VERSION:
        return None
    return payload


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "uri": str(path),
        "byte_count": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _summary(report: Mapping[str, object]) -> str:
    counts = report["counts"]
    assert isinstance(counts, Mapping)
    return (
        "# Frozen prototype reference embeddings\n\n"
        f"- Status: {report['status']}\n"
        f"- Embedded support rows: {counts['embedded']} / {counts['frozen_support']}\n"
        f"- Retryable failures: {counts['retryable_failures']}\n"
        f"- Prototypes: {counts['prototypes']}\n"
        f"- Visual-neighbour edges: {counts['visual_neighbour_edges']}\n"
        "- Semantics: prototype experimental screening evidence only\n"
        "- Provider-supported references are not represented as human verified\n"
    )


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.write_parquet(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", dir=path.parent, encoding="utf-8", delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"artifact hash mismatch for {path}: {actual}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return _SHA256_PREFIX + digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 71 or not text.startswith(_SHA256_PREFIX):
        raise ValueError(f"{field} must be a SHA-256 fingerprint")
    try:
        int(text[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a SHA-256 fingerprint") from exc
    return text


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


__all__ = [
    "PROTOTYPE_EMBEDDING_FAILURES_FILE",
    "PROTOTYPE_EMBEDDING_REPORT_FILE",
    "PROTOTYPE_REFERENCE_EMBEDDINGS_FILE",
    "PROTOTYPE_REFERENCE_PROTOTYPES_FILE",
    "PROTOTYPE_VISUAL_NEIGHBOUR_SPECIES_FILE",
    "PrototypeSupportEmbeddingConfig",
    "PrototypeSupportEmbeddingResult",
    "build_prototype_reference_prototypes",
    "build_prototype_visual_neighbours",
    "prototype_embedding_failures_schema",
    "prototype_reference_embeddings_schema",
    "prototype_reference_prototypes_schema",
    "prototype_visual_neighbour_schema",
    "run_prototype_support_embedding_job",
    "validate_prototype_reference_embeddings",
]
