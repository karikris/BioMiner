"""Durable one-time Flickr full-frame embedding artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot, isclose, isfinite
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.storage.parquet import write_parquet
from biominer.vision.bioclip_input_contract import (
    TARGET_AWARE_VISUAL_MODE,
    bioclip_visual_input_contract,
)
from biominer.vision.full_frame_attention import (
    FULL_FRAME_VISUAL_INPUT_VERSION,
    RAW_FULL_IMAGE_KIND,
    RAW_FULL_IMAGE_TRANSFORMATION_FINGERPRINT,
    TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
    TARGET_FULL_FRAME_PREPROCESSING,
)
from biominer.vision.target_full_frame import (
    TARGET_FULL_FRAME_EMBEDDING_VERSION,
    EmbeddedTargetFullFramePlan,
    FullFrameImageEncoder,
    RawFullFrameEmbedding,
    TargetFullFramePlan,
    encode_target_full_frame_plan,
    full_frame_embedding_id,
)


FLICKR_FULL_FRAME_EMBEDDING_SCHEMA_VERSION = "flickr-full-frame-embedding-v1.0.0"
FLICKR_EMBEDDING_BINDING_SCHEMA_VERSION = "flickr-embedding-binding-v1.0.0"
FLICKR_FULL_FRAME_EMBEDDINGS_FILE = "flickr_full_frame_embeddings.parquet"
FLICKR_EMBEDDING_BINDINGS_FILE = "flickr_embedding_bindings.parquet"

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_BINDING_ID = re.compile(r"flickr-embedding-binding:[0-9a-f]{64}\Z")


def flickr_full_frame_embedding_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "embedding_id": pl.String,
        "embedding_version": pl.String,
        "embedding_fingerprint": pl.String,
        "visual_input_id": pl.String,
        "visual_input_kind": pl.String,
        "visual_input_version": pl.String,
        "raw_image_content_hash": pl.String,
        "transformation_fingerprint": pl.String,
        "spatial_crop_applied": pl.Boolean,
        "model_id": pl.String,
        "model_revision": pl.String,
        "model_fingerprint": pl.String,
        "image_resize_mode": pl.String,
        "preprocessing_contract_fingerprint": pl.String,
        "preprocessing_fingerprint": pl.String,
        "embedding_dimension": pl.UInt32,
        "embedding": pl.List(pl.Float64),
        "embedding_norm": pl.Float64,
        "row_fingerprint": pl.String,
        "embedding_cache_fingerprint": pl.String,
    }


def flickr_embedding_binding_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "binding_id": pl.String,
        "source": pl.String,
        "flickr_photo_id": pl.String,
        "source_record_hash": pl.String,
        "visual_input_contract_version": pl.String,
        "visual_input_contract_fingerprint": pl.String,
        "visual_input_id": pl.String,
        "embedding_id": pl.String,
        "embedding_fingerprint": pl.String,
        "binding_fingerprint": pl.String,
        "binding_set_fingerprint": pl.String,
    }


@dataclass(frozen=True, slots=True)
class FlickrEmbeddingArtifacts:
    embeddings: pl.DataFrame
    photo_bindings: pl.DataFrame


@dataclass(frozen=True, slots=True)
class FlickrEmbeddingPersistenceResult:
    embedded_plan: EmbeddedTargetFullFramePlan
    artifacts: FlickrEmbeddingArtifacts
    embeddings_path: Path
    photo_bindings_path: Path
    embedding_cache_fingerprint: str
    binding_set_fingerprint: str
    visual_inputs_total: int
    photo_bindings_total: int
    cache_hits: int
    cache_misses: int
    encoder_calls: int
    images_encoded: int
    encoder_model_load_count_before: int
    encoder_model_load_count_after: int
    encoder_model_load_count_delta: int


def persist_reusable_flickr_embeddings(
    plan: TargetFullFramePlan,
    *,
    encoder: FullFrameImageEncoder,
    model_id: str,
    model_revision: str,
    model_fingerprint: str,
    preprocessing_fingerprint: str,
    output_dir: str | Path,
) -> FlickrEmbeddingPersistenceResult:
    """Encode cache misses once, merge them, and atomically persist both grains."""

    destination = Path(output_dir)
    embeddings_path = destination / FLICKR_FULL_FRAME_EMBEDDINGS_FILE
    bindings_path = destination / FLICKR_EMBEDDING_BINDINGS_FILE
    if embeddings_path.exists() != bindings_path.exists():
        raise ValueError("Flickr embedding cache is missing one linked artifact")
    existing = (
        load_flickr_embedding_artifacts(destination)
        if embeddings_path.exists()
        else _empty_artifacts()
    )
    normalized_model_id = _required_text(model_id, field="model_id")
    normalized_model_revision = _required_text(model_revision, field="model_revision")
    _validate_encoder_model_identity(
        encoder,
        model_id=normalized_model_id,
        model_revision=normalized_model_revision,
    )
    _validate_cached_model_identity(
        existing.embeddings,
        model_id=normalized_model_id,
        model_revision=normalized_model_revision,
        model_fingerprint=model_fingerprint,
    )
    cached_embeddings = raw_full_frame_embeddings_from_cache(existing.embeddings)
    cached_ids = {item.embedding_id for item in cached_embeddings}
    expected_ids = {
        full_frame_embedding_id(
            visual_input_id=item.visual_input_id,
            model_fingerprint=model_fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
        )
        for item in plan.visual_inputs
    }
    cache_hits = len(expected_ids & cached_ids)
    cache_misses = len(expected_ids - cached_ids)
    model_loads_before = _encoder_model_load_count(encoder)
    embedded = encode_target_full_frame_plan(
        plan,
        encoder=encoder,
        model_fingerprint=model_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
        embedding_cache=cached_embeddings,
    )
    model_loads_after = _encoder_model_load_count(encoder)
    model_load_delta = model_loads_after - model_loads_before
    if model_load_delta < 0:
        raise ValueError("encoder model-load count decreased during embedding")
    if model_load_delta > 1:
        raise ValueError("one Flickr embedding batch loaded the model more than once")

    current = _artifacts_from_embedded_plan(
        plan,
        embedded,
        model_id=normalized_model_id,
        model_revision=normalized_model_revision,
    )
    merged = _merge_artifacts(existing, current)
    validate_flickr_embedding_artifacts(merged)
    written_embeddings = write_parquet(merged.embeddings, embeddings_path)
    written_bindings = write_parquet(merged.photo_bindings, bindings_path)
    durable = load_flickr_embedding_artifacts(destination)
    if not durable.embeddings.equals(merged.embeddings):
        raise ValueError("Flickr embedding Parquet round-trip mismatch")
    if not durable.photo_bindings.equals(merged.photo_bindings):
        raise ValueError("Flickr embedding binding Parquet round-trip mismatch")
    return FlickrEmbeddingPersistenceResult(
        embedded_plan=embedded,
        artifacts=durable,
        embeddings_path=written_embeddings,
        photo_bindings_path=written_bindings,
        embedding_cache_fingerprint=flickr_embedding_cache_fingerprint(
            durable.embeddings
        ),
        binding_set_fingerprint=flickr_embedding_binding_set_fingerprint(
            durable.photo_bindings
        ),
        visual_inputs_total=len(plan.visual_inputs),
        photo_bindings_total=current.photo_bindings.height,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        encoder_calls=int(cache_misses > 0),
        images_encoded=cache_misses,
        encoder_model_load_count_before=model_loads_before,
        encoder_model_load_count_after=model_loads_after,
        encoder_model_load_count_delta=model_load_delta,
    )


def load_flickr_embedding_artifacts(
    directory: str | Path,
) -> FlickrEmbeddingArtifacts:
    source = Path(directory)
    embeddings_path = source / FLICKR_FULL_FRAME_EMBEDDINGS_FILE
    bindings_path = source / FLICKR_EMBEDDING_BINDINGS_FILE
    if not embeddings_path.is_file():
        raise FileNotFoundError(embeddings_path)
    if not bindings_path.is_file():
        raise FileNotFoundError(bindings_path)
    artifacts = FlickrEmbeddingArtifacts(
        embeddings=pl.read_parquet(embeddings_path),
        photo_bindings=pl.read_parquet(bindings_path),
    )
    validate_flickr_embedding_artifacts(artifacts)
    return artifacts


def validate_flickr_embedding_artifacts(
    artifacts: FlickrEmbeddingArtifacts,
) -> None:
    if not isinstance(artifacts, FlickrEmbeddingArtifacts):
        raise TypeError("artifacts must be FlickrEmbeddingArtifacts")
    embeddings = artifacts.embeddings
    bindings = artifacts.photo_bindings
    _require_schema(
        embeddings,
        flickr_full_frame_embedding_schema(),
        label="Flickr embeddings",
    )
    _require_schema(
        bindings,
        flickr_embedding_binding_schema(),
        label="Flickr embedding bindings",
    )
    _validate_embedding_frame(embeddings)
    if not bindings.equals(
        bindings.sort(
            "source",
            "flickr_photo_id",
            "source_record_hash",
            "visual_input_id",
            "embedding_id",
        )
    ):
        raise ValueError("Flickr embedding bindings are not canonically sorted")
    if bindings["binding_id"].n_unique() != bindings.height:
        raise ValueError("Flickr embedding binding IDs are not unique")

    expected_binding_fingerprint = flickr_embedding_binding_set_fingerprint(bindings)
    if bindings.height and set(bindings["binding_set_fingerprint"]) != {
        expected_binding_fingerprint
    }:
        raise ValueError("Flickr embedding binding-set fingerprint mismatch")

    embeddings_by_id = {
        str(row["embedding_id"]): _validated_embedding_row(row)
        for row in embeddings.iter_rows(named=True)
    }
    binding_keys: set[tuple[str, ...]] = set()
    for row in bindings.iter_rows(named=True):
        normalized = _validated_binding_row(row)
        binding_key = tuple(
            str(normalized[field])
            for field in (
                "source",
                "flickr_photo_id",
                "source_record_hash",
                "visual_input_id",
                "embedding_id",
            )
        )
        if binding_key in binding_keys:
            raise ValueError(
                "Flickr embedding bindings duplicate one complete identity"
            )
        binding_keys.add(binding_key)
        try:
            embedding = embeddings_by_id[str(row["embedding_id"])]
        except KeyError as exc:
            raise ValueError("Flickr binding references an unknown embedding") from exc
        if row["visual_input_id"] != embedding["visual_input_id"]:
            raise ValueError("Flickr binding visual input differs from embedding")
        if row["embedding_fingerprint"] != embedding["embedding_fingerprint"]:
            raise ValueError("Flickr binding fingerprint differs from embedding")


def raw_full_frame_embeddings_from_cache(
    frame: pl.DataFrame,
) -> tuple[RawFullFrameEmbedding, ...]:
    _validate_embedding_frame(frame)
    values: list[RawFullFrameEmbedding] = []
    for row in frame.sort("embedding_id").iter_rows(named=True):
        normalized = _validated_embedding_row(row)
        values.append(
            RawFullFrameEmbedding(
                embedding_id=str(normalized["embedding_id"]),
                embedding_version=str(normalized["embedding_version"]),
                embedding_fingerprint=str(normalized["embedding_fingerprint"]),
                visual_input_id=str(normalized["visual_input_id"]),
                visual_input_kind=str(normalized["visual_input_kind"]),
                raw_image_content_hash=str(normalized["raw_image_content_hash"]),
                transformation_fingerprint=str(
                    normalized["transformation_fingerprint"]
                ),
                model_fingerprint=str(normalized["model_fingerprint"]),
                image_resize_mode=str(normalized["image_resize_mode"]),
                preprocessing_contract_fingerprint=str(
                    normalized["preprocessing_contract_fingerprint"]
                ),
                preprocessing_fingerprint=str(normalized["preprocessing_fingerprint"]),
                embedding_dimension=int(normalized["embedding_dimension"]),
                embedding=tuple(normalized["embedding"]),
                embedding_norm=float(normalized["embedding_norm"]),
            )
        )
    return tuple(values)


def flickr_embedding_cache_fingerprint(frame: pl.DataFrame) -> str:
    _require_schema(
        frame,
        flickr_full_frame_embedding_schema(),
        label="Flickr embeddings",
    )
    return canonical_semantic_fingerprint(
        {
            "schema_version": FLICKR_FULL_FRAME_EMBEDDING_SCHEMA_VERSION,
            "row_fingerprints": frame.sort("embedding_id")["row_fingerprint"].to_list(),
        }
    )


def flickr_embedding_binding_set_fingerprint(frame: pl.DataFrame) -> str:
    _require_schema(
        frame,
        flickr_embedding_binding_schema(),
        label="Flickr embedding bindings",
    )
    return canonical_semantic_fingerprint(
        {
            "schema_version": FLICKR_EMBEDDING_BINDING_SCHEMA_VERSION,
            "binding_fingerprints": frame.sort("binding_id")[
                "binding_fingerprint"
            ].to_list(),
        }
    )


def _artifacts_from_embedded_plan(
    plan: TargetFullFramePlan,
    embedded: EmbeddedTargetFullFramePlan,
    *,
    model_id: str,
    model_revision: str,
) -> FlickrEmbeddingArtifacts:
    embedding_by_id = {item.embedding_id: item for item in embedded.embeddings}
    embedding_id_by_visual = {
        item.raw_visual_input_id: item.embedding_id
        for item in embedded.scoring_unit_references
    }
    embedding_rows = [
        _embedding_row(
            item,
            model_id=model_id,
            model_revision=model_revision,
        )
        for item in embedded.embeddings
    ]
    binding_rows_by_id: dict[str, dict[str, object]] = {}
    for unit in plan.scoring_units:
        embedding_id = embedding_id_by_visual[unit.raw_visual_input_id]
        embedding = embedding_by_id[embedding_id]
        base: dict[str, object] = {
            "schema_version": FLICKR_EMBEDDING_BINDING_SCHEMA_VERSION,
            "source": unit.source,
            "flickr_photo_id": unit.flickr_photo_id,
            "source_record_hash": unit.source_record_hash,
            "visual_input_contract_version": plan.visual_input_contract_version,
            "visual_input_contract_fingerprint": (
                plan.visual_input_contract_fingerprint
            ),
            "visual_input_id": unit.raw_visual_input_id,
            "embedding_id": embedding_id,
            "embedding_fingerprint": embedding.embedding_fingerprint,
        }
        fingerprint = canonical_semantic_fingerprint(base)
        row = {
            **base,
            "binding_id": "flickr-embedding-binding:"
            + fingerprint.removeprefix("sha256:"),
            "binding_fingerprint": fingerprint,
            "binding_set_fingerprint": "",
        }
        previous = binding_rows_by_id.setdefault(str(row["binding_id"]), row)
        if previous != row:
            raise ValueError("conflicting Flickr embedding binding identity")
    return _finalize_artifacts(
        embedding_rows,
        list(binding_rows_by_id.values()),
    )


def _embedding_row(
    embedding: RawFullFrameEmbedding,
    *,
    model_id: str,
    model_revision: str,
) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": FLICKR_FULL_FRAME_EMBEDDING_SCHEMA_VERSION,
        "embedding_id": embedding.embedding_id,
        "embedding_version": embedding.embedding_version,
        "embedding_fingerprint": embedding.embedding_fingerprint,
        "visual_input_id": embedding.visual_input_id,
        "visual_input_kind": embedding.visual_input_kind,
        "visual_input_version": FULL_FRAME_VISUAL_INPUT_VERSION,
        "raw_image_content_hash": embedding.raw_image_content_hash,
        "transformation_fingerprint": embedding.transformation_fingerprint,
        "spatial_crop_applied": False,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_fingerprint": embedding.model_fingerprint,
        "image_resize_mode": embedding.image_resize_mode,
        "preprocessing_contract_fingerprint": (
            embedding.preprocessing_contract_fingerprint
        ),
        "preprocessing_fingerprint": embedding.preprocessing_fingerprint,
        "embedding_dimension": embedding.embedding_dimension,
        "embedding": list(embedding.embedding),
        "embedding_norm": embedding.embedding_norm,
    }
    return {
        **base,
        "row_fingerprint": canonical_semantic_fingerprint(base),
        "embedding_cache_fingerprint": "",
    }


def _merge_artifacts(
    existing: FlickrEmbeddingArtifacts,
    current: FlickrEmbeddingArtifacts,
) -> FlickrEmbeddingArtifacts:
    embedding_rows = _merged_rows(
        existing.embeddings,
        current.embeddings,
        identity_field="embedding_id",
        ignored_fields={"embedding_cache_fingerprint"},
    )
    binding_rows = _merged_rows(
        existing.photo_bindings,
        current.photo_bindings,
        identity_field="binding_id",
        ignored_fields={"binding_set_fingerprint"},
    )
    return _finalize_artifacts(embedding_rows, binding_rows)


def _finalize_artifacts(
    embedding_rows: Sequence[Mapping[str, object]],
    binding_rows: Sequence[Mapping[str, object]],
) -> FlickrEmbeddingArtifacts:
    embeddings = pl.DataFrame(
        [dict(row) for row in embedding_rows],
        schema=flickr_full_frame_embedding_schema(),
        orient="row",
        strict=True,
    ).sort("embedding_id")
    cache_fingerprint = flickr_embedding_cache_fingerprint(embeddings)
    embeddings = embeddings.with_columns(
        pl.lit(cache_fingerprint).alias("embedding_cache_fingerprint")
    )
    bindings = pl.DataFrame(
        [dict(row) for row in binding_rows],
        schema=flickr_embedding_binding_schema(),
        orient="row",
        strict=True,
    ).sort(
        "source",
        "flickr_photo_id",
        "source_record_hash",
        "visual_input_id",
        "embedding_id",
    )
    binding_fingerprint = flickr_embedding_binding_set_fingerprint(bindings)
    bindings = bindings.with_columns(
        pl.lit(binding_fingerprint).alias("binding_set_fingerprint")
    )
    return FlickrEmbeddingArtifacts(
        embeddings=embeddings,
        photo_bindings=bindings,
    )


def _empty_artifacts() -> FlickrEmbeddingArtifacts:
    return FlickrEmbeddingArtifacts(
        embeddings=pl.DataFrame(schema=flickr_full_frame_embedding_schema()),
        photo_bindings=pl.DataFrame(schema=flickr_embedding_binding_schema()),
    )


def _merged_rows(
    existing: pl.DataFrame,
    current: pl.DataFrame,
    *,
    identity_field: str,
    ignored_fields: set[str],
) -> list[dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {}
    for raw in (*existing.to_dicts(), *current.to_dicts()):
        row = dict(raw)
        for field in ignored_fields:
            row[field] = ""
        identity = str(row[identity_field])
        previous = by_id.setdefault(identity, row)
        if previous != row:
            raise ValueError(f"conflicting durable {identity_field} identity")
    return list(by_id.values())


def _validated_embedding_row(row: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(row)
    if normalized["schema_version"] != FLICKR_FULL_FRAME_EMBEDDING_SCHEMA_VERSION:
        raise ValueError("unsupported Flickr embedding schema version")
    for field in (
        "embedding_id",
        "embedding_fingerprint",
        "visual_input_id",
        "raw_image_content_hash",
        "transformation_fingerprint",
        "model_fingerprint",
        "preprocessing_contract_fingerprint",
        "preprocessing_fingerprint",
        "row_fingerprint",
        "embedding_cache_fingerprint",
    ):
        _sha256(normalized[field], field=field)
    if normalized["embedding_version"] != TARGET_FULL_FRAME_EMBEDDING_VERSION:
        raise ValueError("unsupported Flickr embedding version")
    if normalized["visual_input_kind"] != RAW_FULL_IMAGE_KIND:
        raise ValueError("Flickr embedding must use the raw full image")
    if normalized["visual_input_version"] != FULL_FRAME_VISUAL_INPUT_VERSION:
        raise ValueError("unsupported Flickr full-frame visual-input version")
    if normalized["spatial_crop_applied"] is not False:
        raise ValueError("Flickr full-frame embedding cannot apply a spatial crop")
    _required_text(normalized["model_id"], field="model_id")
    _required_text(normalized["model_revision"], field="model_revision")
    if (
        normalized["transformation_fingerprint"]
        != RAW_FULL_IMAGE_TRANSFORMATION_FINGERPRINT
    ):
        raise ValueError("Flickr embedding transformation fingerprint mismatch")
    if normalized["image_resize_mode"] != TARGET_FULL_FRAME_IMAGE_RESIZE_MODE:
        raise ValueError("Flickr embedding resize mode mismatch")
    if (
        normalized["preprocessing_contract_fingerprint"]
        != TARGET_FULL_FRAME_PREPROCESSING.fingerprint
    ):
        raise ValueError("Flickr embedding preprocessing contract mismatch")
    expected_visual_input_id = canonical_semantic_fingerprint(
        {
            "raw_image_content_hash": normalized["raw_image_content_hash"],
            "transformation_fingerprint": normalized["transformation_fingerprint"],
            "visual_input_kind": normalized["visual_input_kind"],
            "visual_input_version": normalized["visual_input_version"],
        }
    )
    if normalized["visual_input_id"] != expected_visual_input_id:
        raise ValueError("Flickr embedding visual-input identity mismatch")
    expected_embedding_id = full_frame_embedding_id(
        visual_input_id=str(normalized["visual_input_id"]),
        model_fingerprint=str(normalized["model_fingerprint"]),
        preprocessing_fingerprint=str(normalized["preprocessing_fingerprint"]),
    )
    if normalized["embedding_id"] != expected_embedding_id:
        raise ValueError("Flickr embedding ID mismatch")
    vector = _vector(normalized["embedding"])
    dimension = _positive_int(
        normalized["embedding_dimension"], field="embedding_dimension"
    )
    if dimension != len(vector):
        raise ValueError("Flickr embedding dimension mismatch")
    norm = _finite_float(normalized["embedding_norm"], field="embedding_norm")
    expected_norm = hypot(*vector)
    if expected_norm == 0 or not isfinite(expected_norm):
        raise ValueError("Flickr embedding norm must be finite and non-zero")
    if not isclose(norm, expected_norm, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Flickr embedding norm mismatch")
    expected_embedding_fingerprint = canonical_semantic_fingerprint(
        {
            "embedding": vector,
            "embedding_id": normalized["embedding_id"],
            "embedding_version": normalized["embedding_version"],
        }
    )
    if normalized["embedding_fingerprint"] != expected_embedding_fingerprint:
        raise ValueError("Flickr embedding fingerprint mismatch")
    base = {
        field: normalized[field]
        for field in flickr_full_frame_embedding_schema()
        if field not in {"row_fingerprint", "embedding_cache_fingerprint"}
    }
    if normalized["row_fingerprint"] != canonical_semantic_fingerprint(base):
        raise ValueError("Flickr embedding row fingerprint mismatch")
    normalized["embedding"] = vector
    normalized["embedding_dimension"] = dimension
    normalized["embedding_norm"] = norm
    return normalized


def _validated_binding_row(row: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(row)
    if normalized["schema_version"] != FLICKR_EMBEDDING_BINDING_SCHEMA_VERSION:
        raise ValueError("unsupported Flickr embedding binding schema version")
    if not _BINDING_ID.fullmatch(str(normalized["binding_id"])):
        raise ValueError("invalid Flickr embedding binding ID")
    if normalized["source"] != "flickr":
        raise ValueError("Flickr embedding binding requires source='flickr'")
    _required_text(normalized["flickr_photo_id"], field="flickr_photo_id")
    for field in (
        "source_record_hash",
        "visual_input_contract_fingerprint",
        "visual_input_id",
        "embedding_id",
        "embedding_fingerprint",
        "binding_fingerprint",
        "binding_set_fingerprint",
    ):
        _sha256(normalized[field], field=field)
    input_contract = bioclip_visual_input_contract(TARGET_AWARE_VISUAL_MODE)
    if normalized["visual_input_contract_version"] != input_contract.contract_version:
        raise ValueError("Flickr binding visual-input contract version mismatch")
    if normalized["visual_input_contract_fingerprint"] != input_contract.fingerprint:
        raise ValueError("Flickr binding visual-input contract fingerprint mismatch")
    base = {
        field: normalized[field]
        for field in flickr_embedding_binding_schema()
        if field
        not in {
            "binding_id",
            "binding_fingerprint",
            "binding_set_fingerprint",
        }
    }
    fingerprint = canonical_semantic_fingerprint(base)
    expected_id = "flickr-embedding-binding:" + fingerprint.removeprefix("sha256:")
    if normalized["binding_id"] != expected_id:
        raise ValueError("Flickr embedding binding ID mismatch")
    if normalized["binding_fingerprint"] != fingerprint:
        raise ValueError("Flickr embedding binding fingerprint mismatch")
    return normalized


def _encoder_model_load_count(encoder: object) -> int:
    value = getattr(encoder, "model_load_count", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("encoder model_load_count must be a non-negative integer")
    return value


def _validate_encoder_model_identity(
    encoder: object,
    *,
    model_id: str,
    model_revision: str,
) -> None:
    for attribute, expected in (
        ("model_id", model_id),
        ("model_revision", model_revision),
    ):
        actual = getattr(encoder, attribute, None)
        if actual is not None and str(actual) != expected:
            raise ValueError(f"encoder {attribute} differs from Flickr cache contract")


def _validate_cached_model_identity(
    frame: pl.DataFrame,
    *,
    model_id: str,
    model_revision: str,
    model_fingerprint: str,
) -> None:
    for row in frame.filter(pl.col("model_fingerprint") == model_fingerprint).iter_rows(
        named=True
    ):
        if row["model_id"] != model_id or row["model_revision"] != model_revision:
            raise ValueError(
                "Flickr cache model fingerprint is bound to another model identity"
            )


def _validate_embedding_frame(frame: pl.DataFrame) -> None:
    _require_schema(
        frame,
        flickr_full_frame_embedding_schema(),
        label="Flickr embeddings",
    )
    if not frame.equals(frame.sort("embedding_id")):
        raise ValueError("Flickr embeddings are not canonically sorted")
    if frame["embedding_id"].n_unique() != frame.height:
        raise ValueError("Flickr embedding IDs are not unique")
    expected_fingerprint = flickr_embedding_cache_fingerprint(frame)
    if frame.height and set(frame["embedding_cache_fingerprint"]) != {
        expected_fingerprint
    }:
        raise ValueError("Flickr embedding cache fingerprint mismatch")
    for row in frame.iter_rows(named=True):
        _validated_embedding_row(row)


def _require_schema(
    frame: pl.DataFrame,
    schema: dict[str, pl.DataType],
    *,
    label: str,
) -> None:
    if frame.schema != schema:
        raise ValueError(f"{label} schema mismatch")


def _vector(value: object) -> tuple[float, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError("Flickr embedding vector must be a sequence")
    vector = tuple(
        _finite_float(item, field="embedding vector value") for item in value
    )
    if not vector:
        raise ValueError("Flickr embedding vector cannot be empty")
    return vector


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{field} must be a canonical SHA-256 fingerprint")
    return text


__all__ = [
    "FLICKR_EMBEDDING_BINDINGS_FILE",
    "FLICKR_EMBEDDING_BINDING_SCHEMA_VERSION",
    "FLICKR_FULL_FRAME_EMBEDDINGS_FILE",
    "FLICKR_FULL_FRAME_EMBEDDING_SCHEMA_VERSION",
    "FlickrEmbeddingArtifacts",
    "FlickrEmbeddingPersistenceResult",
    "flickr_embedding_binding_schema",
    "flickr_embedding_binding_set_fingerprint",
    "flickr_embedding_cache_fingerprint",
    "flickr_full_frame_embedding_schema",
    "load_flickr_embedding_artifacts",
    "persist_reusable_flickr_embeddings",
    "raw_full_frame_embeddings_from_cache",
    "validate_flickr_embedding_artifacts",
]
