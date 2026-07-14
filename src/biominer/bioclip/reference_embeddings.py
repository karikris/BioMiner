from __future__ import annotations

from array import array
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import logging
from math import isfinite, sqrt
import os
from pathlib import Path
import re
import shutil
import struct
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Protocol
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from PIL import Image
import polars as pl

from biominer.common.semantic_hash import (
    canonical_semantic_bytes,
    canonical_semantic_fingerprint,
)
from biominer.detection.detector_base import DecodedImage
from biominer.references.readiness import (
    PERMITTING_READINESS_STATUSES,
    REFERENCE_MODEL_INPUT_IDENTITY_SCHEMA_VERSION,
    REFERENCE_SUPPORT_SPLITS,
    ReferenceBankReadinessPermit,
    ReferenceModelInputIdentity,
    reference_support_manifest_fingerprint,
    reference_support_manifest_schema,
    reference_route_dimensions,
    reference_support_split_leakage,
    validate_reference_support_manifest,
)
from biominer.references.schemas import REFERENCE_VIEWS
from biominer.reports.flickr_fetch import current_git_sha
from biominer.storage.cloud import CloudStorage
from biominer.storage.parquet import write_parquet
from biominer.storage.uri import join_uri
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    FULL_FRAME_VISUAL_INPUT_VERSION,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
    RAW_FULL_IMAGE_TRANSFORMATION_FINGERPRINT,
    TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
    FullFrameAttentionVariant,
    TargetPreprocessingContract,
    decoded_image_content_hash,
)
from biominer.workstore.base import WorkStore


REFERENCE_EMBEDDINGS_SCHEMA_VERSION = "reference-embeddings-v2.0.0"
REFERENCE_EMBEDDINGS_REPORT_SCHEMA_VERSION = "reference-embeddings-report-v1.0.0"
REFERENCE_EMBEDDINGS_SUMMARY_SCHEMA_VERSION = "reference-embeddings-summary-v1.0.0"
REFERENCE_EMBEDDINGS_FILE = "reference_embeddings.parquet"
REFERENCE_EMBEDDINGS_REPORT_FILE = "reference_embeddings_report.json"
REFERENCE_EMBEDDINGS_SUMMARY_FILE = "reference_embeddings_summary.md"
REFERENCE_EMBEDDINGS_MANIFEST_FILE = "manifest.json"
REFERENCE_EMBEDDINGS_MANIFEST_SCHEMA_VERSION = "biominer-artifact-manifest-v1"
REFERENCE_EMBEDDINGS_CHECKPOINT_SCHEMA_VERSION = (
    "reference-embeddings-checkpoint-v2.0.0"
)
REFERENCE_EMBEDDINGS_CHECKPOINT_STATE_FILE = "checkpoint.json"
REFERENCE_EMBEDDINGS_CHECKPOINT_PARTS_DIR = "parts"
REFERENCE_EMBEDDINGS_CHECKPOINT_LOCK_SUFFIX = ".writer.lock"
REFERENCE_EMBEDDING_DTYPE = "float32"
REFERENCE_EMBEDDING_NORMALIZATION_POLICY = "l2-unit-normalize-before-float32-persist-v1"
REFERENCE_EMBEDDING_MODEL_FINGERPRINT_SCHEMA_VERSION = "reference-embedding-model-v1"

_VISUAL_INPUT_KIND_ORDER = {
    RAW_FULL_IMAGE_KIND: 0,
    FOCUSED_FULL_FRAME_KIND: 1,
    MASKED_FULL_FRAME_KIND: 2,
    MULTI_OBJECT_FULL_FRAME_KIND: 3,
}
_SUPPORT_MANIFEST_SORT = (
    "accepted_taxon_key",
    "geo_cluster_id",
    "route",
    "support_split",
    "reference_media_id",
)
_EMBEDDING_GRAIN = (
    "reference_media_id",
    "support_row_fingerprint",
    "model_input_fingerprint",
    "input_contract_version",
    "visual_input_id",
    "visual_input_kind",
    "raw_image_content_hash",
    "image_content_hash",
    "transformation_version",
    "transformation_fingerprint",
    "model_input_schema_version",
    "model_name",
    "model_version",
    "model_id",
    "model_revision",
    "model_weights_sha256",
    "preprocessing_version",
    "preprocessing_fingerprint",
    "preprocessing_attestation_fingerprint",
)
_EMBEDDING_SUPPORT_PROJECTION = (
    "registry_version",
    "reference_bank_version",
    "reference_media_id",
    "reference_observation_id",
    "source_snapshot_version",
    "duplicate_group_id",
    "reference_bank_fingerprint",
    "support_manifest_fingerprint",
    "support_row_fingerprint",
    "accepted_taxon_key",
    "scientific_name",
    "geo_cluster_id",
    "life_stage",
    "visual_domain",
    "view",
    "route",
    "source_image_sha256",
    "support_split",
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{7,64}\Z")
_UNIT_NORM_TOLERANCE = 1e-5
_NORM_ROUNDTRIP_TOLERANCE = 1e-12
_LOGGER = logging.getLogger(__name__)


class ReferenceImageEmbeddingScorer(Protocol):
    model_id: str
    model_revision: str
    model_weights_sha256: str | None
    image_resize_mode: str | None
    effective_image_resize_mode: str | None
    open_clip_version: str | None
    open_clip_config_sha256: str | None
    preprocessing_version: str | None
    preprocessing_config: Mapping[str, object] | None
    preprocessing_fingerprint: str | None
    last_image_content_hashes: Sequence[str]

    def embed_image_paths(
        self,
        image_paths: Sequence[Path],
    ) -> list[list[float]]: ...


class ReferenceEmbeddingCheckpointBusyError(RuntimeError):
    """Raised when another process owns a reference checkpoint writer lock."""


@dataclass(frozen=True, slots=True)
class ReferenceVisualInput:
    reference_media_id: str
    source_image_path: Path
    image_path: Path
    visual_input_id: str
    visual_input_kind: str
    raw_image_content_hash: str
    image_content_hash: str
    transformation_version: str
    transformation_policy_fingerprint: str
    transformation_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_media_id",
            _required_text(self.reference_media_id, field="reference_media_id"),
        )
        object.__setattr__(self, "source_image_path", Path(self.source_image_path))
        object.__setattr__(self, "image_path", Path(self.image_path))
        object.__setattr__(
            self,
            "visual_input_id",
            _sha256(self.visual_input_id, field="visual_input_id"),
        )
        kind = _required_text(self.visual_input_kind, field="visual_input_kind")
        if kind not in _VISUAL_INPUT_KIND_ORDER:
            raise ValueError(f"unsupported reference visual input kind: {kind}")
        object.__setattr__(self, "visual_input_kind", kind)
        object.__setattr__(
            self,
            "raw_image_content_hash",
            _sha256(
                self.raw_image_content_hash,
                field="raw_image_content_hash",
            ),
        )
        object.__setattr__(
            self,
            "image_content_hash",
            _sha256(self.image_content_hash, field="image_content_hash"),
        )
        object.__setattr__(
            self,
            "transformation_version",
            _required_text(
                self.transformation_version,
                field="transformation_version",
            ),
        )
        object.__setattr__(
            self,
            "transformation_policy_fingerprint",
            _sha256(
                self.transformation_policy_fingerprint,
                field="transformation_policy_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "transformation_fingerprint",
            _sha256(
                self.transformation_fingerprint,
                field="transformation_fingerprint",
            ),
        )

    @classmethod
    def from_variant(
        cls,
        *,
        reference_media_id: str,
        source_image_path: str | Path,
        image_path: str | Path,
        variant: FullFrameAttentionVariant,
    ) -> ReferenceVisualInput:
        if not isinstance(variant, FullFrameAttentionVariant):
            raise TypeError("variant must be a FullFrameAttentionVariant")
        return cls(
            reference_media_id=reference_media_id,
            source_image_path=Path(source_image_path),
            image_path=Path(image_path),
            visual_input_id=variant.visual_input_id,
            visual_input_kind=variant.visual_input_kind,
            raw_image_content_hash=variant.raw_image_content_hash,
            image_content_hash=variant.visual_content_hash,
            transformation_version=variant.transformation_version,
            transformation_policy_fingerprint=(
                variant.transformation_policy_fingerprint
            ),
            transformation_fingerprint=variant.transformation_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class _PreparedVisualInput:
    visual_input: ReferenceVisualInput
    support_row: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _PreprocessingAttestation:
    open_clip_version: str
    open_clip_config_sha256: str
    preprocessing_version: str
    preprocessing_config_json: str
    preprocessing_fingerprint: str


@dataclass(frozen=True, slots=True)
class _LoadedEmbeddingCheckpoint:
    frame: pl.DataFrame | None = None
    embedding_created_at: datetime | None = None
    embedding_dimension: int | None = None
    model_weights_sha256: str | None = None
    preprocessing_attestation: _PreprocessingAttestation | None = None
    generation: int = 0
    parts_fingerprint: str | None = None


def reference_embeddings_schema(
    embedding_dimension: int,
) -> dict[str, pl.DataType]:
    dimension = _positive_dimension(embedding_dimension)
    return {
        "schema_version": pl.String,
        "registry_version": pl.String,
        "reference_bank_version": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "source_snapshot_version": pl.String,
        "review_decision_ids": pl.List(pl.String),
        "duplicate_group_id": pl.String,
        "readiness_sha256": pl.String,
        "reference_bank_fingerprint": pl.String,
        "support_manifest_fingerprint": pl.String,
        "model_input_fingerprint": pl.String,
        "input_contract_version": pl.String,
        "support_row_fingerprint": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "geo_cluster_id": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "view": pl.String,
        "route": pl.String,
        "source_object_uri": pl.String,
        "source_image_sha256": pl.String,
        "source_object_fingerprint": pl.String,
        "visual_input_id": pl.String,
        "visual_input_kind": pl.String,
        "raw_image_content_hash": pl.String,
        "image_content_hash": pl.String,
        "transformation_version": pl.String,
        "transformation_policy_fingerprint": pl.String,
        "transformation_fingerprint": pl.String,
        "model_input_schema_version": pl.String,
        "model_name": pl.String,
        "model_version": pl.String,
        "model_id": pl.String,
        "model_revision": pl.String,
        "model_checkpoint_uri": pl.String,
        "model_weights_sha256": pl.String,
        "model_checkpoint_hash": pl.String,
        "model_fingerprint": pl.String,
        "preprocessing_version": pl.String,
        "preprocessing_fingerprint": pl.String,
        "open_clip_version": pl.String,
        "open_clip_config_sha256": pl.String,
        "preprocessing_attestation_version": pl.String,
        "preprocessing_config_json": pl.String,
        "preprocessing_attestation_fingerprint": pl.String,
        "embedding_dimension": pl.UInt32,
        "embedding": pl.Array(pl.Float32, dimension),
        "embedding_norm": pl.Float64,
        "support_split": pl.String,
        "embedding_created_at": pl.Datetime("us", "UTC"),
        "embedding_fingerprint": pl.String,
    }


def build_reference_embeddings(
    support_manifest: pl.DataFrame,
    visual_inputs: Sequence[ReferenceVisualInput],
    *,
    readiness_permit: ReferenceBankReadinessPermit,
    scorer: ReferenceImageEmbeddingScorer,
    preprocessing_contract: TargetPreprocessingContract | None = None,
    batch_size: int = 64,
    embedding_created_at: datetime | None = None,
    embedding_cache: pl.DataFrame | str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = True,
    run_id: str | None = None,
) -> pl.DataFrame:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("reference embedding batch_size must be a positive integer")
    contract = preprocessing_contract or TargetPreprocessingContract()
    if not isinstance(contract, TargetPreprocessingContract):
        raise TypeError("preprocessing_contract must be a TargetPreprocessingContract")
    if not isinstance(resume, bool):
        raise TypeError("reference embedding resume must be a boolean")
    requested_created_at = (
        _utc_datetime(embedding_created_at, field="embedding_created_at")
        if embedding_created_at is not None
        else None
    )
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    effective_run_id = _required_text(
        run_id or f"reference-embeddings-{uuid4().hex}",
        field="run_id",
    )
    started_at = datetime.now(UTC)
    _log_event(
        "reference_embedding_build_started",
        command="bioclip.build_reference_embeddings",
        run_id=effective_run_id,
        pid=os.getpid(),
        batch_size=batch_size,
        visual_input_count=len(visual_inputs),
        checkpoint_dir=str(checkpoint_root) if checkpoint_root is not None else None,
        resume=resume,
        started_at=started_at.isoformat(),
    )
    checkpoint_lock: BinaryIO | None = None
    try:
        manifest = _validated_support_manifest(
            support_manifest,
            readiness_permit=readiness_permit,
        )
        eligible = manifest.filter(pl.col("support_eligible"))
        if eligible.is_empty():
            raise ValueError("reference support manifest has no eligible support rows")
        model_id = _required_scorer_text(scorer, "model_id")
        model_revision = _required_scorer_text(scorer, "model_revision")
        if readiness_permit.model_name.removeprefix("hf-hub:") != model_id:
            raise ValueError("reference readiness model name does not match scorer")
        if readiness_permit.model_revision != model_revision:
            raise ValueError("reference readiness model revision does not match scorer")
        if readiness_permit.input_contract_version != FULL_FRAME_VISUAL_INPUT_VERSION:
            raise ValueError(
                "reference readiness visual input contract does not match builder"
            )
        if readiness_permit.preprocessing_version != contract.version:
            raise ValueError(
                "reference readiness preprocessing version does not match contract"
            )
        if readiness_permit.preprocessing_contract_fingerprint != contract.fingerprint:
            raise ValueError(
                "reference readiness preprocessing contract fingerprint mismatch"
            )
        model_input_identity = readiness_permit.model_input_identity()
        if model_input_identity.fingerprint != readiness_permit.model_input_fingerprint:
            raise ValueError("reference readiness model input fingerprint mismatch")
        if getattr(scorer, "image_resize_mode", None) != (
            TARGET_FULL_FRAME_IMAGE_RESIZE_MODE
        ):
            raise ValueError(
                "reference embeddings require BioCLIP longest-side resize mode"
            )
        ensure_attestation = getattr(scorer, "ensure_model_attestation", None)
        if callable(ensure_attestation):
            ensure_attestation()
        scorer_weights_sha256, scorer_attestation = _validated_scorer_runtime_identity(
            scorer,
            contract=contract,
            readiness_permit=readiness_permit,
            model_id=model_id,
            model_revision=model_revision,
        )
        prepared = _prepare_visual_inputs(eligible, visual_inputs)
        checkpoint_fingerprint = _checkpoint_build_fingerprint(
            prepared,
            readiness_permit=readiness_permit,
            model_id=model_id,
            model_revision=model_revision,
            preprocessing_contract=contract,
        )
        checkpoint_lock = _acquire_embedding_checkpoint_lock(
            checkpoint_root,
            build_fingerprint=checkpoint_fingerprint,
        )
        checkpoint = _load_embedding_checkpoint(
            checkpoint_root,
            expected_build_fingerprint=checkpoint_fingerprint,
            requested_created_at=requested_created_at,
            resume=resume,
        )
        resumed_frame = checkpoint.frame
        resumed_row_count = resumed_frame.height if resumed_frame is not None else 0
        new_frames: list[pl.DataFrame] = []
        new_row_count = 0
        created_at = checkpoint.embedding_created_at or (
            requested_created_at or datetime.now(UTC)
        )
        dimension = checkpoint.embedding_dimension
        frozen_weights_sha256 = checkpoint.model_weights_sha256 or scorer_weights_sha256
        if frozen_weights_sha256 != scorer_weights_sha256:
            raise ValueError(
                "reference embedding checkpoint model weights do not match scorer"
            )
        frozen_attestation = checkpoint.preprocessing_attestation or scorer_attestation
        if frozen_attestation != scorer_attestation:
            raise ValueError(
                "reference embedding checkpoint preprocessing does not match scorer"
            )
        checkpoint_generation = checkpoint.generation
        checkpoint_parts_fingerprint = (
            checkpoint.parts_fingerprint or _json_fingerprint([])
        )
        embedding_vectors_by_cache_key, cached_dimension = (
            _load_reference_embedding_vector_cache(
                embedding_cache,
                readiness_permit=readiness_permit,
                model_id=model_id,
                model_revision=model_revision,
                preprocessing_contract=contract,
                preprocessing_attestation=scorer_attestation,
            )
        )
        if dimension is None:
            dimension = cached_dimension
        elif cached_dimension is not None and cached_dimension != dimension:
            raise ValueError(
                "reference embedding cache and checkpoint dimensions do not match"
            )
        resumed_frame = _rebind_resumed_checkpoint_provenance(
            resumed_frame,
            prepared,
            readiness_permit=readiness_permit,
            model_id=model_id,
            model_revision=model_revision,
            preprocessing_contract=contract,
        )
        if resumed_frame is not None:
            _merge_reference_embedding_vectors_into_cache(
                embedding_vectors_by_cache_key,
                resumed_frame,
            )
        completed = {
            (str(support_row_fingerprint), str(visual_input_id))
            for support_row_fingerprint, visual_input_id in (
                resumed_frame.select(
                    "support_row_fingerprint",
                    "visual_input_id",
                ).iter_rows()
                if resumed_frame is not None
                else ()
            )
        }
        expected_keys = frozenset(_prepared_checkpoint_key(item) for item in prepared)
        if len(expected_keys) != len(prepared):
            raise AssertionError("reference prepared checkpoint keys are not unique")
        unknown_completed = sorted(completed - expected_keys)
        if unknown_completed:
            raise ValueError(
                "reference embedding checkpoint contains unknown completed inputs"
            )
        pending = [
            item for item in prepared if _prepared_checkpoint_key(item) not in completed
        ]
        _validate_visual_input_files(pending)
        pending_by_cache_key: dict[
            tuple[str, str, str, str, str, str],
            list[_PreparedVisualInput],
        ] = {}
        for item in pending:
            pending_by_cache_key.setdefault(
                _prepared_embedding_cache_key(
                    item,
                    input_contract_version=readiness_permit.input_contract_version,
                    model_id=model_id,
                    model_revision=model_revision,
                    preprocessing_version=contract.version,
                    model_input_fingerprint=readiness_permit.model_input_fingerprint,
                ),
                [],
            ).append(item)
        pending_groups = list(pending_by_cache_key.values())
        batch_count = 0
        for start in range(0, len(pending_groups), batch_size):
            groups = pending_groups[start : start + batch_size]
            batch = [
                group[0]
                for group in groups
                if _prepared_embedding_cache_key(
                    group[0],
                    input_contract_version=readiness_permit.input_contract_version,
                    model_id=model_id,
                    model_revision=model_revision,
                    preprocessing_version=contract.version,
                    model_input_fingerprint=readiness_permit.model_input_fingerprint,
                )
                not in embedding_vectors_by_cache_key
            ]
            paths = [item.visual_input.image_path for item in batch]
            vectors = scorer.embed_image_paths(paths) if paths else []
            batch_count += 1
            if batch:
                worker_hashes = tuple(
                    _sha256(value, field="worker image content hash")
                    for value in getattr(scorer, "last_image_content_hashes", ())
                )
                expected_hashes = tuple(
                    item.visual_input.image_content_hash for item in batch
                )
                if worker_hashes != expected_hashes:
                    raise ValueError(
                        "reference image scorer embedded content does not match "
                        "visual input hashes"
                    )
                if len(vectors) != len(batch):
                    raise ValueError(
                        f"reference image scorer returned {len(vectors)} rows for "
                        f"{len(batch)} images"
                    )
                weights_sha256, attestation = _validated_scorer_runtime_identity(
                    scorer,
                    contract=contract,
                    readiness_permit=readiness_permit,
                    model_id=model_id,
                    model_revision=model_revision,
                )
                if weights_sha256 != frozen_weights_sha256:
                    raise ValueError(
                        "reference image scorer model weights SHA-256 changed between batches"
                    )
                if attestation != frozen_attestation:
                    raise ValueError(
                        "reference image scorer preprocessing attestation changed "
                        "between batches"
                    )
                for item, raw_vector in zip(batch, vectors, strict=True):
                    key = _prepared_embedding_cache_key(
                        item,
                        input_contract_version=readiness_permit.input_contract_version,
                        model_id=model_id,
                        model_revision=model_revision,
                        preprocessing_version=contract.version,
                        model_input_fingerprint=(
                            readiness_permit.model_input_fingerprint
                        ),
                    )
                    vector, norm = _stored_unit_vector(raw_vector)
                    embedding_vectors_by_cache_key[key] = (vector, norm)
            batch_rows: list[dict[str, object]] = []
            for group in groups:
                cache_key = _prepared_embedding_cache_key(
                    group[0],
                    input_contract_version=readiness_permit.input_contract_version,
                    model_id=model_id,
                    model_revision=model_revision,
                    preprocessing_version=contract.version,
                    model_input_fingerprint=readiness_permit.model_input_fingerprint,
                )
                vector, norm = embedding_vectors_by_cache_key[cache_key]
                if dimension is None:
                    dimension = len(vector)
                elif len(vector) != dimension:
                    raise ValueError(
                        "reference image scorer returned mixed embedding dimensions"
                    )
                for item in group:
                    row = _embedding_row(
                        item,
                        vector=vector,
                        embedding_norm=norm,
                        embedding_dimension=dimension,
                        model_id=model_id,
                        model_revision=model_revision,
                        model_weights_sha256=frozen_weights_sha256,
                        preprocessing_contract=contract,
                        preprocessing_attestation=frozen_attestation,
                        readiness_permit=readiness_permit,
                        embedding_created_at=created_at,
                    )
                    row["embedding_fingerprint"] = _embedding_row_fingerprint(row)
                    batch_rows.append(row)
            batch_frame = _sort_embedding_frame(
                pl.DataFrame(
                    batch_rows,
                    schema=reference_embeddings_schema(dimension),
                    orient="row",
                    strict=True,
                )
            )
            new_frames.append(batch_frame)
            new_row_count += batch_frame.height
            if checkpoint_root is not None:
                checkpoint_generation, checkpoint_parts_fingerprint = (
                    _write_embedding_checkpoint_batch(
                        checkpoint_root,
                        batch_frame=batch_frame,
                        completed_row_count=resumed_row_count + new_row_count,
                        build_fingerprint=checkpoint_fingerprint,
                        embedding_created_at=created_at,
                        embedding_dimension=dimension,
                        model_weights_sha256=frozen_weights_sha256,
                        preprocessing_attestation=frozen_attestation,
                        expected_generation=checkpoint_generation,
                        expected_parts_fingerprint=checkpoint_parts_fingerprint,
                    )
                )
            _log_event(
                "reference_embedding_batch_completed",
                command="bioclip.build_reference_embeddings",
                run_id=effective_run_id,
                batch_number=batch_count,
                batch_rows=len(batch_rows),
                unique_model_inputs=len(batch),
                completed_rows=resumed_row_count + new_row_count,
                total_rows=len(prepared),
                checkpoint_dir=(
                    str(checkpoint_root) if checkpoint_root is not None else None
                ),
                error_count=0,
            )
        if (
            dimension is None
            or frozen_weights_sha256 is None
            or frozen_attestation is None
        ):
            raise AssertionError("reference embedding build produced no vectors")
        frames = [resumed_frame] if resumed_frame is not None else []
        frames.extend(new_frames)
        frame = pl.concat(frames, rechunk=True)
        frame = _sort_embedding_frame(frame)
        actual_keys = {
            (str(support_row_fingerprint), str(visual_input_id))
            for support_row_fingerprint, visual_input_id in frame.select(
                "support_row_fingerprint",
                "visual_input_id",
            ).iter_rows()
        }
        if frame.height != len(prepared) or actual_keys != expected_keys:
            raise ValueError("reference embedding output coverage mismatch")
        validate_reference_embeddings(
            frame,
            expected_model_id=model_id,
            expected_model_revision=model_revision,
            expected_model_weights_sha256=frozen_weights_sha256,
            expected_preprocessing_version=contract.version,
            expected_preprocessing_fingerprint=contract.fingerprint,
            expected_preprocessing_attestation_fingerprint=(
                frozen_attestation.preprocessing_fingerprint
            ),
            expected_model_input_fingerprint=(readiness_permit.model_input_fingerprint),
            expected_input_contract_version=(readiness_permit.input_contract_version),
        )
    except Exception as exc:
        _log_event(
            "reference_embedding_build_failed",
            command="bioclip.build_reference_embeddings",
            run_id=effective_run_id,
            pid=os.getpid(),
            error_type=type(exc).__name__,
            error=str(exc),
            ended_at=datetime.now(UTC).isoformat(),
        )
        raise
    finally:
        _release_embedding_checkpoint_lock(checkpoint_lock)
    _log_event(
        "reference_embedding_build_completed",
        command="bioclip.build_reference_embeddings",
        run_id=effective_run_id,
        pid=os.getpid(),
        row_count=frame.height,
        embedding_dimension=dimension,
        batch_count=batch_count,
        resumed_row_count=resumed_row_count,
        error_count=0,
        elapsed_seconds=(datetime.now(UTC) - started_at).total_seconds(),
    )
    return frame


def decoded_image_file_content_hash(path: str | Path) -> str:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            rgb.load()
            width, height = rgb.size
            decoded = DecodedImage(
                width=int(width),
                height=int(height),
                mode="RGB",
                data=rgb.tobytes(),
                source_uri=str(image_path),
            )
    except Exception as exc:  # noqa: BLE001 - include path in decode failures.
        raise ValueError(
            f"reference visual input cannot be decoded: {image_path}"
        ) from exc
    return decoded_image_content_hash(decoded)


def validate_reference_embeddings(
    frame: pl.DataFrame,
    *,
    expected_model_id: str | None = None,
    expected_model_revision: str | None = None,
    expected_model_weights_sha256: str | None = None,
    expected_preprocessing_version: str | None = None,
    expected_preprocessing_fingerprint: str | None = None,
    expected_preprocessing_attestation_fingerprint: str | None = None,
    expected_model_input_fingerprint: str | None = None,
    expected_input_contract_version: str | None = None,
    require_raw_full_image: bool = True,
) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("reference embeddings must be a Polars DataFrame")
    expected_columns = list(reference_embeddings_schema(1))
    if frame.columns != expected_columns:
        raise ValueError("reference embeddings physical schema mismatch")
    if frame.is_empty():
        raise ValueError("reference embeddings artifact must not be empty")
    dimensions = frame["embedding_dimension"].unique().to_list()
    if len(dimensions) != 1:
        raise ValueError("reference embeddings artifact has mixed dimensions")
    dimension = _positive_dimension(dimensions[0])
    if dict(frame.schema) != reference_embeddings_schema(dimension):
        raise ValueError("reference embeddings physical schema mismatch")
    if not frame.equals(_sort_embedding_frame(frame)):
        raise ValueError("reference embeddings rows are not deterministically sorted")
    if frame.select(list(_EMBEDDING_GRAIN)).unique().height != frame.height:
        raise ValueError("reference embeddings contain duplicate grain rows")
    if (
        frame.select("support_row_fingerprint", "visual_input_id").unique().height
        != frame.height
    ):
        raise ValueError("reference embeddings contain duplicate checkpoint keys")
    if frame["embedding_fingerprint"].n_unique() != frame.height:
        raise ValueError("reference embeddings contain duplicate fingerprints")

    model_id = _require_single_value(frame, "model_id", expected_model_id)
    _require_single_value(frame, "registry_version", None)
    _require_single_value(frame, "reference_bank_version", None)
    model_input_schema_version = _require_single_value(
        frame,
        "model_input_schema_version",
        REFERENCE_MODEL_INPUT_IDENTITY_SCHEMA_VERSION,
    )
    model_name = _require_single_value(frame, "model_name", None)
    if model_id != model_name.removeprefix("hf-hub:"):
        raise ValueError("reference embeddings model ID does not match model name")
    model_version = _require_single_value(frame, "model_version", None)
    model_revision = _require_single_value(
        frame,
        "model_revision",
        expected_model_revision,
    )
    weights_sha256 = _require_single_value(
        frame,
        "model_weights_sha256",
        expected_model_weights_sha256,
    )
    _sha256(weights_sha256, field="model_weights_sha256")
    _require_single_value(frame, "model_checkpoint_hash", weights_sha256)
    preprocessing_version = _require_single_value(
        frame,
        "preprocessing_version",
        expected_preprocessing_version,
    )
    preprocessing_fingerprint = _require_single_value(
        frame,
        "preprocessing_fingerprint",
        expected_preprocessing_fingerprint,
    )
    _sha256(preprocessing_fingerprint, field="preprocessing_fingerprint")
    for field in (
        "readiness_sha256",
        "reference_bank_fingerprint",
        "support_manifest_fingerprint",
    ):
        _sha256(_require_single_value(frame, field, None), field=field)
    model_input_fingerprint = _require_single_value(
        frame,
        "model_input_fingerprint",
        expected_model_input_fingerprint,
    )
    _sha256(model_input_fingerprint, field="model_input_fingerprint")
    model_fingerprint = _require_single_value(
        frame,
        "model_fingerprint",
        _reference_embedding_model_fingerprint(
            model_input_fingerprint=model_input_fingerprint,
            embedding_dimension=dimension,
        ),
    )
    _sha256(model_fingerprint, field="model_fingerprint")
    open_clip_version = _require_single_value(frame, "open_clip_version", None)
    open_clip_config_sha256 = _require_single_value(
        frame,
        "open_clip_config_sha256",
        None,
    )
    _sha256(open_clip_config_sha256, field="open_clip_config_sha256")
    input_contract_version = _require_single_value(
        frame,
        "input_contract_version",
        expected_input_contract_version,
    )
    model_checkpoint_uri = _require_single_value(
        frame,
        "model_checkpoint_uri",
        None,
    )
    _validate_absolute_uri(model_checkpoint_uri, field="model_checkpoint_uri")
    _require_single_value(frame, "preprocessing_attestation_version", None)
    attestation_fingerprint = _require_single_value(
        frame,
        "preprocessing_attestation_fingerprint",
        expected_preprocessing_attestation_fingerprint,
    )
    _sha256(
        attestation_fingerprint,
        field="preprocessing_attestation_fingerprint",
    )
    model_input_identity = ReferenceModelInputIdentity(
        schema_version=model_input_schema_version,
        model_name=model_name,
        model_version=model_version,
        model_revision=model_revision,
        checkpoint_uri=model_checkpoint_uri,
        checkpoint_sha256=weights_sha256,
        open_clip_version=open_clip_version,
        open_clip_config_sha256=open_clip_config_sha256,
        preprocessing_version=preprocessing_version,
        preprocessing_contract_fingerprint=preprocessing_fingerprint,
        preprocessing_attestation_fingerprint=attestation_fingerprint,
        input_contract_version=input_contract_version,
    )
    if model_input_identity.fingerprint != model_input_fingerprint:
        raise ValueError("reference embeddings model input fingerprint mismatch")
    _preprocessing_attestation_from_row(frame.row(0, named=True))

    split_by_identity: dict[tuple[str, str], str] = {}
    raw_input_counts: Counter[str] = Counter()
    support_projection_by_fingerprint: dict[str, tuple[object, ...]] = {}
    support_fingerprint_by_media: dict[str, str] = {}
    vector_by_cache_key: dict[
        tuple[str, str, str, str, str, str], tuple[tuple[float, ...], float]
    ] = {}
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != REFERENCE_EMBEDDINGS_SCHEMA_VERSION:
            raise ValueError("unsupported reference embeddings schema version")
        for field in (
            "reference_media_id",
            "reference_observation_id",
            "source_snapshot_version",
            "duplicate_group_id",
            "accepted_taxon_key",
            "scientific_name",
            "geo_cluster_id",
            "life_stage",
            "visual_domain",
            "view",
            "route",
            "source_object_uri",
            "input_contract_version",
            "transformation_version",
            "model_input_schema_version",
            "model_name",
            "model_version",
            "model_id",
            "model_revision",
            "preprocessing_version",
            "open_clip_version",
            "preprocessing_attestation_version",
            "preprocessing_config_json",
        ):
            _required_text(row[field], field=field)
        decision_ids = row["review_decision_ids"]
        if not isinstance(decision_ids, list) or not decision_ids:
            raise ValueError("review_decision_ids must be a non-empty string list")
        for decision_id in decision_ids:
            _required_text(decision_id, field="review_decision_ids")
        if decision_ids != sorted(set(decision_ids)):
            raise ValueError("review_decision_ids must be sorted and unique")
        _validate_absolute_uri(row["source_object_uri"], field="source_object_uri")
        for field in (
            "readiness_sha256",
            "reference_bank_fingerprint",
            "support_manifest_fingerprint",
            "model_input_fingerprint",
            "support_row_fingerprint",
            "source_image_sha256",
            "source_object_fingerprint",
            "visual_input_id",
            "raw_image_content_hash",
            "image_content_hash",
            "transformation_policy_fingerprint",
            "transformation_fingerprint",
            "model_weights_sha256",
            "preprocessing_fingerprint",
            "open_clip_config_sha256",
            "preprocessing_attestation_fingerprint",
        ):
            _sha256(row[field], field=field)
        if _preprocessing_attestation_from_row(row).preprocessing_fingerprint != (
            attestation_fingerprint
        ):
            raise ValueError("reference embeddings preprocessing attestation mismatch")
        kind = str(row["visual_input_kind"])
        if kind not in _VISUAL_INPUT_KIND_ORDER:
            raise ValueError(f"unsupported reference visual input kind: {kind}")
        if kind == RAW_FULL_IMAGE_KIND:
            raw_input_counts[str(row["reference_media_id"])] += 1
        split = str(row["support_split"])
        if split not in REFERENCE_SUPPORT_SPLITS:
            raise ValueError(f"unsupported reference support split: {split}")
        route = str(row["route"])
        if row["view"] not in REFERENCE_VIEWS:
            raise ValueError(f"unsupported reference view: {row['view']}")
        expected_life_stage, expected_visual_domain = reference_route_dimensions(route)
        if (
            row["life_stage"] != expected_life_stage
            or row["visual_domain"] != expected_visual_domain
        ):
            raise ValueError(
                "reference embedding route dimensions mismatch: "
                f"{route} requires {expected_life_stage}/{expected_visual_domain}"
            )
        if row["visual_input_id"] != _expected_visual_input_id_from_values(
            visual_input_kind=kind,
            raw_image_content_hash=str(row["raw_image_content_hash"]),
            image_content_hash=str(row["image_content_hash"]),
            transformation_fingerprint=str(row["transformation_fingerprint"]),
        ):
            raise ValueError("reference visual input identity fingerprint mismatch")
        if kind == RAW_FULL_IMAGE_KIND:
            if row["image_content_hash"] != row["raw_image_content_hash"]:
                raise ValueError("raw reference visual input content hash mismatch")
            if (
                row["transformation_policy_fingerprint"]
                != RAW_FULL_IMAGE_TRANSFORMATION_FINGERPRINT
                or row["transformation_fingerprint"]
                != RAW_FULL_IMAGE_TRANSFORMATION_FINGERPRINT
            ):
                raise ValueError("raw reference visual input transformation mismatch")
        for field in (
            "reference_media_id",
            "reference_observation_id",
            "duplicate_group_id",
            "source_image_sha256",
            "raw_image_content_hash",
            "image_content_hash",
        ):
            identity = (field, str(row[field]))
            previous_split = split_by_identity.setdefault(identity, split)
            if previous_split != split:
                raise ValueError(f"reference {field} crosses support splits")
        support_fingerprint = str(row["support_row_fingerprint"])
        support_projection = tuple(
            tuple(row[field]) if isinstance(row[field], list) else row[field]
            for field in _EMBEDDING_SUPPORT_PROJECTION
        )
        previous_projection = support_projection_by_fingerprint.setdefault(
            support_fingerprint,
            support_projection,
        )
        if previous_projection != support_projection:
            raise ValueError(
                "reference embedding support projection mismatch for support row"
            )
        media_id = str(row["reference_media_id"])
        previous_support_fingerprint = support_fingerprint_by_media.setdefault(
            media_id,
            support_fingerprint,
        )
        if previous_support_fingerprint != support_fingerprint:
            raise ValueError("reference embedding media maps to multiple support rows")
        vector = tuple(float(value) for value in row["embedding"])
        if len(vector) != dimension or any(not isfinite(value) for value in vector):
            raise ValueError(
                "reference embedding vector dimension or values are invalid"
            )
        norm = sqrt(sum(value * value for value in vector))
        if not isfinite(norm) or norm <= 0:
            raise ValueError("reference embedding vector has non-zero norm violation")
        if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
            raise ValueError("reference embedding vector is not unit-normalized")
        stored_norm = float(row["embedding_norm"])
        if not isfinite(stored_norm) or abs(norm - stored_norm) > (
            _NORM_ROUNDTRIP_TOLERANCE
        ):
            raise ValueError("reference embedding norm mismatch")
        cache_key = _embedding_cache_key_from_row(row)
        cache_value = (vector, stored_norm)
        previous_cache_value = vector_by_cache_key.setdefault(
            cache_key,
            cache_value,
        )
        if previous_cache_value != cache_value:
            raise ValueError(
                "reference embeddings have conflicting vectors for one "
                "content and model identity"
            )
        if int(row["embedding_dimension"]) != dimension:
            raise ValueError("reference embedding dimension mismatch")
        _utc_datetime(row["embedding_created_at"], field="embedding_created_at")
        _sha256(row["embedding_fingerprint"], field="embedding_fingerprint")
        if row["embedding_fingerprint"] != _embedding_row_fingerprint(row):
            raise ValueError("reference embedding fingerprint mismatch")
    if require_raw_full_image:
        media_ids = {str(value) for value in frame["reference_media_id"].to_list()}
        invalid_raw_counts = sorted(
            media_id for media_id in media_ids if raw_input_counts[media_id] != 1
        )
        if invalid_raw_counts:
            raise ValueError(
                "reference embeddings require exactly one raw full-image row per media: "
                + ", ".join(invalid_raw_counts[:10])
            )


def reference_embeddings_artifact_fingerprint(
    frame: pl.DataFrame,
    *,
    require_raw_full_image: bool = True,
) -> str:
    validate_reference_embeddings(
        frame,
        require_raw_full_image=require_raw_full_image,
    )
    digest = hashlib.sha256()
    for fingerprint in frame["embedding_fingerprint"].to_list():
        encoded = str(fingerprint).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return "sha256:" + digest.hexdigest()


def write_reference_embeddings(
    frame: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    validate_reference_embeddings(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= REFERENCE_EMBEDDINGS_FILE
    written = write_parquet(frame, destination, overwrite=overwrite)
    loaded = _read_reference_embeddings_parquet(written)
    if not frame.equals(loaded):
        raise ValueError("reference embeddings Parquet round-trip mismatch")
    return written


def load_reference_embeddings(
    path: str | Path,
    *,
    expected_model_id: str | None = None,
    expected_model_revision: str | None = None,
    expected_model_weights_sha256: str | None = None,
    expected_preprocessing_version: str | None = None,
    expected_preprocessing_fingerprint: str | None = None,
    expected_preprocessing_attestation_fingerprint: str | None = None,
    expected_model_input_fingerprint: str | None = None,
    expected_input_contract_version: str | None = None,
) -> pl.DataFrame:
    source = Path(path)
    publication_directory: Path | None = None
    if source.is_dir():
        publication_directory = source
        source /= REFERENCE_EMBEDDINGS_FILE
    elif source.name == REFERENCE_EMBEDDINGS_FILE:
        publication_directory = source.parent
    if (
        publication_directory is not None
        and not (publication_directory / REFERENCE_EMBEDDINGS_MANIFEST_FILE).is_file()
    ):
        raise ValueError("reference embeddings publication manifest is missing")
    frame = _read_reference_embeddings_parquet(
        source,
        expected_model_id=expected_model_id,
        expected_model_revision=expected_model_revision,
        expected_model_weights_sha256=expected_model_weights_sha256,
        expected_preprocessing_version=expected_preprocessing_version,
        expected_preprocessing_fingerprint=expected_preprocessing_fingerprint,
        expected_preprocessing_attestation_fingerprint=(
            expected_preprocessing_attestation_fingerprint
        ),
        expected_model_input_fingerprint=expected_model_input_fingerprint,
        expected_input_contract_version=expected_input_contract_version,
    )
    if publication_directory is not None:
        _validate_local_reference_embeddings_publication(
            publication_directory,
            frame,
        )
    return frame


def _read_reference_embeddings_parquet(
    source: str | Path,
    *,
    expected_model_id: str | None = None,
    expected_model_revision: str | None = None,
    expected_model_weights_sha256: str | None = None,
    expected_preprocessing_version: str | None = None,
    expected_preprocessing_fingerprint: str | None = None,
    expected_preprocessing_attestation_fingerprint: str | None = None,
    expected_model_input_fingerprint: str | None = None,
    expected_input_contract_version: str | None = None,
) -> pl.DataFrame:
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pl.read_parquet(source)
    validate_reference_embeddings(
        frame,
        expected_model_id=expected_model_id,
        expected_model_revision=expected_model_revision,
        expected_model_weights_sha256=expected_model_weights_sha256,
        expected_preprocessing_version=expected_preprocessing_version,
        expected_preprocessing_fingerprint=expected_preprocessing_fingerprint,
        expected_preprocessing_attestation_fingerprint=(
            expected_preprocessing_attestation_fingerprint
        ),
        expected_model_input_fingerprint=expected_model_input_fingerprint,
        expected_input_contract_version=expected_input_contract_version,
    )
    return frame


def _validate_local_reference_embeddings_publication(
    directory: Path,
    frame: pl.DataFrame,
) -> None:
    manifest_path = directory / REFERENCE_EMBEDDINGS_MANIFEST_FILE
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "reference embeddings publication manifest is invalid"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("reference embeddings publication manifest is invalid")
    required = {
        "schema_version",
        "artifact_set_name",
        "artifact_set_version",
        "status",
        "run_id",
        "job_name",
        "stage",
        "registry_version",
        "git_sha",
        "started_at",
        "ended_at",
        "effective_configuration",
        "files",
        "dependency_fingerprints",
        "source_snapshot_versions",
        "qa",
        "metrics",
    }
    if set(manifest) != required:
        raise ValueError("reference embeddings publication manifest schema mismatch")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list) or len(manifest_files) != 3:
        raise ValueError("reference embeddings publication manifest files are invalid")
    published_uris: dict[str, str] = {}
    for record in manifest_files:
        if not isinstance(record, Mapping):
            raise ValueError(
                "reference embeddings publication manifest files are invalid"
            )
        name = _required_text(record.get("name"), field="manifest file name")
        if name in published_uris:
            raise ValueError(
                "reference embeddings publication manifest files are invalid"
            )
        published_uris[name] = _required_text(
            record.get("uri"),
            field=f"manifest {name} uri",
        )
    if set(published_uris) != {
        REFERENCE_EMBEDDINGS_FILE,
        REFERENCE_EMBEDDINGS_REPORT_FILE,
        REFERENCE_EMBEDDINGS_SUMMARY_FILE,
    }:
        raise ValueError("reference embeddings publication manifest files are invalid")
    artifact_fingerprint = reference_embeddings_artifact_fingerprint(frame)
    run_id = _required_text(manifest.get("run_id"), field="manifest run_id")
    git_sha = _provenance_git_sha(manifest.get("git_sha"))
    job_name = _required_text(manifest.get("job_name"), field="manifest job_name")
    stage = _required_text(manifest.get("stage"), field="manifest stage")
    started_at = _utc_datetime(
        datetime.fromisoformat(str(manifest.get("started_at"))),
        field="manifest started_at",
    )
    ended_at = _utc_datetime(
        datetime.fromisoformat(str(manifest.get("ended_at"))),
        field="manifest ended_at",
    )
    if ended_at < started_at:
        raise ValueError("reference embeddings publication time range is invalid")
    report_path = directory / REFERENCE_EMBEDDINGS_REPORT_FILE
    summary_path = directory / REFERENCE_EMBEDDINGS_SUMMARY_FILE
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("reference embeddings publication report is invalid") from exc
    if not isinstance(report, Mapping):
        raise ValueError("reference embeddings publication report is invalid")
    command = _required_text(report.get("command"), field="publication command")
    if command == "bioclip.publish_reference_embeddings":
        if job_name != "reference-embeddings-local" or stage != "frozen-support":
            raise ValueError("reference embeddings local publication identity mismatch")
        if "worker_id" in report:
            raise ValueError("reference embeddings local publication identity mismatch")
        worker_id = None
        embeddings_uri = (directory / REFERENCE_EMBEDDINGS_FILE).resolve().as_uri()
        record_uris: Mapping[str, str] | None = None
    elif command == "bioclip.publish_reference_embeddings_to_cloud":
        worker_id = _required_text(report.get("worker_id"), field="report worker_id")
        embeddings_uri = published_uris[REFERENCE_EMBEDDINGS_FILE]
        record_uris = published_uris
    else:
        raise ValueError("reference embeddings publication command is invalid")
    embeddings_path = directory / REFERENCE_EMBEDDINGS_FILE
    _validate_existing_reference_embedding_report(
        report,
        frame=frame,
        artifact_uri=embeddings_uri,
        artifact_byte_count=embeddings_path.stat().st_size,
        artifact_fingerprint=artifact_fingerprint,
        artifact_sha256=_file_sha256(embeddings_path),
        run_id=run_id,
        git_sha=git_sha,
        command=command,
        worker_id=worker_id,
    )
    try:
        summary = summary_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("reference embeddings publication summary is missing") from exc
    if summary != _publication_markdown(report):
        raise ValueError("reference embeddings publication summary/report mismatch")
    file_records = _local_reference_embeddings_file_records(
        frame,
        content_directory=directory,
        published_directory=directory,
        published_uris=record_uris,
        report=report,
        artifact_fingerprint=artifact_fingerprint,
    )
    expected = _reference_embeddings_manifest_payload(
        frame,
        file_records=file_records,
        artifact_fingerprint=artifact_fingerprint,
        registry_version=str(frame["registry_version"][0]),
        run_id=run_id,
        job_name=job_name,
        stage=stage,
        git_sha=git_sha,
        started_at=started_at,
        ended_at=ended_at,
    )
    if manifest != expected:
        raise ValueError("reference embeddings publication manifest metadata mismatch")


def _local_reference_embeddings_file_records(
    frame: pl.DataFrame,
    *,
    content_directory: Path,
    published_directory: Path,
    published_uris: Mapping[str, str] | None = None,
    report: Mapping[str, object],
    artifact_fingerprint: str,
) -> list[dict[str, object]]:
    embeddings_path = content_directory / REFERENCE_EMBEDDINGS_FILE
    report_path = content_directory / REFERENCE_EMBEDDINGS_REPORT_FILE
    summary_path = content_directory / REFERENCE_EMBEDDINGS_SUMMARY_FILE
    for path in (embeddings_path, report_path, summary_path):
        if not path.is_file():
            raise ValueError(
                f"reference embeddings publication file is missing: {path.name}"
            )

    def published_uri(name: str) -> str:
        if published_uris is not None:
            return _required_text(
                published_uris.get(name),
                field=f"published {name} uri",
            )
        return (published_directory / name).resolve().as_uri()

    return [
        {
            "name": REFERENCE_EMBEDDINGS_FILE,
            "uri": published_uri(REFERENCE_EMBEDDINGS_FILE),
            "byte_count": embeddings_path.stat().st_size,
            "row_count": frame.height,
            "physical_schema_version": REFERENCE_EMBEDDINGS_SCHEMA_VERSION,
            "sha256": _file_sha256(embeddings_path),
            "semantic_fingerprint": artifact_fingerprint,
            "primary_key": ["support_row_fingerprint", "visual_input_id"],
            "sort_order": [
                "accepted_taxon_key",
                "route",
                "geo_cluster_id",
                "support_split",
                "reference_media_id",
                "visual_input_kind",
                "image_content_hash",
                "transformation_version",
                "transformation_fingerprint",
                "visual_input_id",
            ],
        },
        {
            "name": REFERENCE_EMBEDDINGS_REPORT_FILE,
            "uri": published_uri(REFERENCE_EMBEDDINGS_REPORT_FILE),
            "byte_count": report_path.stat().st_size,
            "row_count": 1,
            "physical_schema_version": REFERENCE_EMBEDDINGS_REPORT_SCHEMA_VERSION,
            "sha256": _file_sha256(report_path),
            "semantic_fingerprint": _publication_report_semantic_fingerprint(report),
            "primary_key": ["run_id"],
            "sort_order": ["run_id"],
        },
        {
            "name": REFERENCE_EMBEDDINGS_SUMMARY_FILE,
            "uri": published_uri(REFERENCE_EMBEDDINGS_SUMMARY_FILE),
            "byte_count": summary_path.stat().st_size,
            "row_count": None,
            "physical_schema_version": REFERENCE_EMBEDDINGS_SUMMARY_SCHEMA_VERSION,
            "sha256": _file_sha256(summary_path),
            "semantic_fingerprint": _publication_summary_semantic_fingerprint(report),
            "primary_key": [],
            "sort_order": [],
        },
    ]


def publish_reference_embeddings(
    frame: pl.DataFrame,
    output_dir: str | Path,
    *,
    run_id: str | None = None,
    git_sha: str | None = None,
) -> dict[str, Path]:
    validate_reference_embeddings(frame)
    directory = Path(output_dir)
    if directory.suffix:
        raise ValueError("reference embeddings publication output must be a directory")
    if directory.exists():
        raise FileExistsError(directory)
    effective_run_id = _required_text(
        run_id or f"reference-embeddings-publish-{uuid4().hex}",
        field="run_id",
    )
    effective_git_sha = _provenance_git_sha(git_sha)
    started_at = datetime.now(UTC)
    staging = directory.parent / f".{directory.name}.{uuid4().hex}.tmp"
    _log_event(
        "reference_embedding_publication_started",
        command="bioclip.publish_reference_embeddings",
        run_id=effective_run_id,
        pid=os.getpid(),
        output_dir=str(directory),
        row_count=frame.height,
        started_at=started_at.isoformat(),
    )
    try:
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir(parents=False, exist_ok=False)
        embeddings_path = write_reference_embeddings(
            frame,
            staging / REFERENCE_EMBEDDINGS_FILE,
            overwrite=False,
        )
        ended_at = datetime.now(UTC)
        embeddings_uri = (directory / REFERENCE_EMBEDDINGS_FILE).resolve().as_uri()
        report_path = staging / REFERENCE_EMBEDDINGS_REPORT_FILE
        summary_path = staging / REFERENCE_EMBEDDINGS_SUMMARY_FILE
        report = _publication_report(
            frame,
            artifact_uri=embeddings_uri,
            artifact_byte_count=embeddings_path.stat().st_size,
            artifact_sha256=_file_sha256(embeddings_path),
            run_id=effective_run_id,
            git_sha=effective_git_sha,
            pid=os.getpid(),
            started_at=started_at,
            ended_at=ended_at,
        )
        report_path.write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        summary_path.write_text(
            _publication_markdown(report),
            encoding="utf-8",
        )
        artifact_fingerprint = reference_embeddings_artifact_fingerprint(frame)
        file_records = _local_reference_embeddings_file_records(
            frame,
            content_directory=staging,
            published_directory=directory,
            report=report,
            artifact_fingerprint=artifact_fingerprint,
        )
        manifest = _reference_embeddings_manifest_payload(
            frame,
            file_records=file_records,
            artifact_fingerprint=artifact_fingerprint,
            registry_version=str(frame["registry_version"][0]),
            run_id=effective_run_id,
            job_name="reference-embeddings-local",
            stage="frozen-support",
            git_sha=effective_git_sha,
            started_at=started_at,
            ended_at=ended_at,
        )
        manifest_path = staging / REFERENCE_EMBEDDINGS_MANIFEST_FILE
        _write_json_atomically(manifest_path, manifest)
        for path in (embeddings_path, report_path, summary_path, manifest_path):
            _fsync_file(path)
        _fsync_directory(staging)
        if directory.exists():
            raise FileExistsError(directory)
        staging.replace(directory)
        _fsync_directory(directory.parent)
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        _log_event(
            "reference_embedding_publication_failed",
            command="bioclip.publish_reference_embeddings",
            run_id=effective_run_id,
            pid=os.getpid(),
            output_dir=str(directory),
            error_type=type(exc).__name__,
            error=str(exc),
            ended_at=datetime.now(UTC).isoformat(),
        )
        raise
    _log_event(
        "reference_embedding_publication_completed",
        command="bioclip.publish_reference_embeddings",
        run_id=effective_run_id,
        pid=os.getpid(),
        output_dir=str(directory),
        row_count=frame.height,
        artifact_bytes=report["artifact"]["byte_count"],
        ended_at=ended_at.isoformat(),
    )
    return {
        "embeddings": directory / REFERENCE_EMBEDDINGS_FILE,
        "report": directory / REFERENCE_EMBEDDINGS_REPORT_FILE,
        "summary": directory / REFERENCE_EMBEDDINGS_SUMMARY_FILE,
        "manifest": directory / REFERENCE_EMBEDDINGS_MANIFEST_FILE,
    }


def publish_reference_embeddings_to_cloud(
    frame: pl.DataFrame,
    output_prefix: str,
    *,
    storage: CloudStorage,
    workstore: WorkStore,
    job_name: str,
    stage: str,
    registry_version: str,
    run_id: str,
    worker_id: str,
    git_sha: str | None = None,
) -> dict[str, str]:
    prefix = _validate_absolute_uri(output_prefix, field="output_prefix")
    manifest_uri = join_uri(prefix, REFERENCE_EMBEDDINGS_MANIFEST_FILE)
    with workstore.publication_lock(manifest_uri):
        return _publish_reference_embeddings_to_cloud_locked(
            frame,
            output_prefix,
            storage=storage,
            workstore=workstore,
            job_name=job_name,
            stage=stage,
            registry_version=registry_version,
            run_id=run_id,
            worker_id=worker_id,
            git_sha=git_sha,
        )


def _publish_reference_embeddings_to_cloud_locked(
    frame: pl.DataFrame,
    output_prefix: str,
    *,
    storage: CloudStorage,
    workstore: WorkStore,
    job_name: str,
    stage: str,
    registry_version: str,
    run_id: str,
    worker_id: str,
    git_sha: str | None,
) -> dict[str, str]:
    validate_reference_embeddings(frame)
    prefix = _validate_absolute_uri(output_prefix, field="output_prefix")
    effective_job_name = _required_text(job_name, field="job_name")
    effective_stage = _required_text(stage, field="stage")
    effective_registry_version = _required_text(
        registry_version,
        field="registry_version",
    )
    if set(frame["registry_version"].to_list()) != {effective_registry_version}:
        raise ValueError("reference embeddings registry version mismatch")
    effective_run_id = _required_text(run_id, field="run_id")
    effective_worker_id = _required_text(worker_id, field="worker_id")
    effective_git_sha = _provenance_git_sha(git_sha)
    embeddings_uri = join_uri(prefix, REFERENCE_EMBEDDINGS_FILE)
    report_uri = join_uri(prefix, REFERENCE_EMBEDDINGS_REPORT_FILE)
    summary_uri = join_uri(prefix, REFERENCE_EMBEDDINGS_SUMMARY_FILE)
    manifest_uri = join_uri(prefix, REFERENCE_EMBEDDINGS_MANIFEST_FILE)
    artifact_fingerprint = reference_embeddings_artifact_fingerprint(frame)
    started_at = datetime.now(UTC)
    _log_event(
        "reference_embedding_cloud_publication_started",
        command="bioclip.publish_reference_embeddings_to_cloud",
        run_id=effective_run_id,
        worker_id=effective_worker_id,
        artifact_uri=embeddings_uri,
        row_count=frame.height,
        started_at=started_at.isoformat(),
    )
    try:
        if storage.exists(manifest_uri):
            manifest = storage.read_json(manifest_uri)
            artifact_record = _manifest_file_record(
                manifest,
                REFERENCE_EMBEDDINGS_FILE,
            )
            if storage.file_size(embeddings_uri) != _non_negative_integer(
                artifact_record.get("byte_count"),
                field="manifest file byte_count",
            ):
                raise ValueError(
                    "reference embeddings cloud manifest file size mismatch"
                )
            if storage.file_sha256(embeddings_uri) != _sha256(
                artifact_record.get("sha256"),
                field="manifest file SHA-256",
            ):
                raise ValueError(
                    "reference embeddings cloud manifest checksum mismatch"
                )
            durable_frame = storage.read_parquet(embeddings_uri)
            validate_reference_embeddings(durable_frame)
            if (
                reference_embeddings_artifact_fingerprint(durable_frame)
                != artifact_fingerprint
            ):
                raise ValueError(
                    "reference embeddings cloud artifact fingerprint mismatch"
                )
            _validate_reference_embeddings_cloud_manifest(
                manifest,
                storage=storage,
                frame=durable_frame,
                manifest_uri=manifest_uri,
                embeddings_uri=embeddings_uri,
                report_uri=report_uri,
                summary_uri=summary_uri,
                artifact_fingerprint=artifact_fingerprint,
                registry_version=effective_registry_version,
                run_id=effective_run_id,
                job_name=effective_job_name,
                stage=effective_stage,
                git_sha=effective_git_sha,
            )
            artifact_byte_count = int(artifact_record["byte_count"])
            artifact_sha256 = str(artifact_record["sha256"])
            ended_at = datetime.fromisoformat(str(manifest["ended_at"]))
            _register_reference_embeddings_cloud_shard(
                durable_frame,
                workstore=workstore,
                artifact_fingerprint=artifact_fingerprint,
                artifact_uri=embeddings_uri,
                artifact_sha256=artifact_sha256,
                artifact_byte_count=artifact_byte_count,
                report_uri=report_uri,
                summary_uri=summary_uri,
                manifest_uri=manifest_uri,
                job_name=effective_job_name,
                stage=effective_stage,
                registry_version=effective_registry_version,
                run_id=effective_run_id,
                worker_id=effective_worker_id,
            )
            _log_event(
                "reference_embedding_cloud_publication_completed",
                command="bioclip.publish_reference_embeddings_to_cloud",
                run_id=effective_run_id,
                worker_id=effective_worker_id,
                artifact_uri=embeddings_uri,
                row_count=durable_frame.height,
                artifact_bytes=artifact_byte_count,
                reused=True,
                ended_at=ended_at.isoformat(),
            )
            return {
                "embeddings": embeddings_uri,
                "report": report_uri,
                "summary": summary_uri,
                "manifest": manifest_uri,
            }
        if not storage.exists(embeddings_uri):
            try:
                storage.write_parquet_shard(
                    embeddings_uri,
                    frame,
                    overwrite=False,
                )
            except FileExistsError:
                pass
        loaded = storage.read_parquet(embeddings_uri)
        validate_reference_embeddings(loaded)
        if reference_embeddings_artifact_fingerprint(loaded) != artifact_fingerprint:
            raise FileExistsError(
                "reference embeddings cloud artifact already exists with "
                "different content"
            )
        artifact_byte_count = storage.file_size(embeddings_uri)
        artifact_sha256 = storage.file_sha256(embeddings_uri)
        if storage.exists(report_uri):
            report = storage.read_json(report_uri)
            _validate_existing_reference_embedding_report(
                report,
                frame=loaded,
                artifact_uri=embeddings_uri,
                artifact_byte_count=artifact_byte_count,
                artifact_fingerprint=artifact_fingerprint,
                artifact_sha256=artifact_sha256,
                run_id=effective_run_id,
                git_sha=effective_git_sha,
                command="bioclip.publish_reference_embeddings_to_cloud",
                worker_id=_required_text(
                    report.get("worker_id"),
                    field="report worker_id",
                ),
            )
            ended_at = _utc_datetime(
                datetime.fromisoformat(str(report["ended_at"])),
                field="reference embedding report ended_at",
            )
        else:
            ended_at = datetime.now(UTC)
            report = _publication_report(
                loaded,
                artifact_uri=embeddings_uri,
                artifact_byte_count=artifact_byte_count,
                artifact_sha256=artifact_sha256,
                run_id=effective_run_id,
                git_sha=effective_git_sha,
                pid=os.getpid(),
                started_at=started_at,
                ended_at=ended_at,
                command="bioclip.publish_reference_embeddings_to_cloud",
                worker_id=effective_worker_id,
            )
        _write_cloud_json_immutable(storage, report_uri, report)
        report = storage.read_json(report_uri)
        _validate_existing_reference_embedding_report(
            report,
            frame=loaded,
            artifact_uri=embeddings_uri,
            artifact_byte_count=artifact_byte_count,
            artifact_fingerprint=artifact_fingerprint,
            artifact_sha256=artifact_sha256,
            run_id=effective_run_id,
            git_sha=effective_git_sha,
            command="bioclip.publish_reference_embeddings_to_cloud",
            worker_id=_required_text(
                report.get("worker_id"),
                field="report worker_id",
            ),
        )
        _write_cloud_text_immutable(
            storage,
            summary_uri,
            _publication_markdown(report),
            content_type="text/markdown; charset=utf-8",
        )
        manifest = _reference_embeddings_cloud_manifest(
            loaded,
            storage=storage,
            embeddings_uri=embeddings_uri,
            report_uri=report_uri,
            summary_uri=summary_uri,
            artifact_fingerprint=artifact_fingerprint,
            registry_version=effective_registry_version,
            run_id=effective_run_id,
            job_name=effective_job_name,
            stage=effective_stage,
            git_sha=str(report["git_sha"]),
            started_at=_utc_datetime(
                datetime.fromisoformat(str(report["started_at"])),
                field="reference embedding report started_at",
            ),
            ended_at=ended_at,
        )
        try:
            _write_cloud_json_immutable(storage, manifest_uri, manifest)
        except FileExistsError:
            manifest = storage.read_json(manifest_uri)
        _validate_reference_embeddings_cloud_manifest(
            manifest,
            storage=storage,
            frame=loaded,
            manifest_uri=manifest_uri,
            embeddings_uri=embeddings_uri,
            report_uri=report_uri,
            summary_uri=summary_uri,
            artifact_fingerprint=artifact_fingerprint,
            registry_version=effective_registry_version,
            run_id=effective_run_id,
            job_name=effective_job_name,
            stage=effective_stage,
            git_sha=effective_git_sha,
        )
        _register_reference_embeddings_cloud_shard(
            loaded,
            workstore=workstore,
            artifact_fingerprint=artifact_fingerprint,
            artifact_uri=embeddings_uri,
            artifact_sha256=artifact_sha256,
            artifact_byte_count=artifact_byte_count,
            report_uri=report_uri,
            summary_uri=summary_uri,
            manifest_uri=manifest_uri,
            job_name=effective_job_name,
            stage=effective_stage,
            registry_version=effective_registry_version,
            run_id=effective_run_id,
            worker_id=effective_worker_id,
        )
    except Exception as exc:
        _log_event(
            "reference_embedding_cloud_publication_failed",
            command="bioclip.publish_reference_embeddings_to_cloud",
            run_id=effective_run_id,
            worker_id=effective_worker_id,
            artifact_uri=embeddings_uri,
            error_type=type(exc).__name__,
            error=str(exc),
            ended_at=datetime.now(UTC).isoformat(),
        )
        raise
    _log_event(
        "reference_embedding_cloud_publication_completed",
        command="bioclip.publish_reference_embeddings_to_cloud",
        run_id=effective_run_id,
        worker_id=effective_worker_id,
        artifact_uri=embeddings_uri,
        row_count=loaded.height,
        artifact_bytes=artifact_byte_count,
        ended_at=ended_at.isoformat(),
    )
    return {
        "embeddings": embeddings_uri,
        "report": report_uri,
        "summary": summary_uri,
        "manifest": manifest_uri,
    }


def _register_reference_embeddings_cloud_shard(
    frame: pl.DataFrame,
    *,
    workstore: WorkStore,
    artifact_fingerprint: str,
    artifact_uri: str,
    artifact_sha256: str,
    artifact_byte_count: int,
    report_uri: str,
    summary_uri: str,
    manifest_uri: str,
    job_name: str,
    stage: str,
    registry_version: str,
    run_id: str,
    worker_id: str,
) -> None:
    shard_id = _json_fingerprint(
        {
            "artifact_uri": artifact_uri,
            "job_name": job_name,
            "run_id": run_id,
            "stage": stage,
        }
    ).removeprefix("sha256:")
    workstore.register_shard(
        shard_id=shard_id,
        job_name=job_name,
        registry_version=registry_version,
        stage=stage,
        run_id=run_id,
        worker_id=worker_id,
        uri=artifact_uri,
        checksum=artifact_sha256,
        row_count=frame.height,
        byte_count=artifact_byte_count,
        metadata={
            "artifact_fingerprint": artifact_fingerprint,
            "manifest_uri": manifest_uri,
            "model_input_fingerprint": str(frame["model_input_fingerprint"][0]),
            "reference_bank_fingerprint": str(frame["reference_bank_fingerprint"][0]),
            "report_uri": report_uri,
            "summary_uri": summary_uri,
        },
    )


def _reference_embeddings_cloud_manifest(
    frame: pl.DataFrame,
    *,
    storage: CloudStorage,
    embeddings_uri: str,
    report_uri: str,
    summary_uri: str,
    artifact_fingerprint: str,
    registry_version: str,
    run_id: str,
    job_name: str,
    stage: str,
    git_sha: str,
    started_at: datetime,
    ended_at: datetime,
) -> dict[str, object]:
    report = storage.read_json(report_uri)
    file_records = [
        {
            "name": REFERENCE_EMBEDDINGS_FILE,
            "uri": embeddings_uri,
            "byte_count": storage.file_size(embeddings_uri),
            "row_count": frame.height,
            "physical_schema_version": REFERENCE_EMBEDDINGS_SCHEMA_VERSION,
            "sha256": storage.file_sha256(embeddings_uri),
            "semantic_fingerprint": artifact_fingerprint,
            "primary_key": ["support_row_fingerprint", "visual_input_id"],
            "sort_order": [
                "accepted_taxon_key",
                "route",
                "geo_cluster_id",
                "support_split",
                "reference_media_id",
                "visual_input_kind",
                "image_content_hash",
                "transformation_version",
                "transformation_fingerprint",
                "visual_input_id",
            ],
        },
        {
            "name": REFERENCE_EMBEDDINGS_REPORT_FILE,
            "uri": report_uri,
            "byte_count": storage.file_size(report_uri),
            "row_count": 1,
            "physical_schema_version": REFERENCE_EMBEDDINGS_REPORT_SCHEMA_VERSION,
            "sha256": storage.file_sha256(report_uri),
            "semantic_fingerprint": _publication_report_semantic_fingerprint(report),
            "primary_key": ["run_id"],
            "sort_order": ["run_id"],
        },
        {
            "name": REFERENCE_EMBEDDINGS_SUMMARY_FILE,
            "uri": summary_uri,
            "byte_count": storage.file_size(summary_uri),
            "row_count": None,
            "physical_schema_version": REFERENCE_EMBEDDINGS_SUMMARY_SCHEMA_VERSION,
            "sha256": storage.file_sha256(summary_uri),
            "semantic_fingerprint": _publication_summary_semantic_fingerprint(report),
            "primary_key": [],
            "sort_order": [],
        },
    ]
    return _reference_embeddings_manifest_payload(
        frame,
        file_records=file_records,
        artifact_fingerprint=artifact_fingerprint,
        registry_version=registry_version,
        run_id=run_id,
        job_name=job_name,
        stage=stage,
        git_sha=git_sha,
        started_at=started_at,
        ended_at=ended_at,
    )


def _reference_embeddings_manifest_payload(
    frame: pl.DataFrame,
    *,
    file_records: Sequence[Mapping[str, object]],
    artifact_fingerprint: str,
    registry_version: str,
    run_id: str,
    job_name: str,
    stage: str,
    git_sha: str,
    started_at: datetime,
    ended_at: datetime,
) -> dict[str, object]:
    return {
        "schema_version": REFERENCE_EMBEDDINGS_MANIFEST_SCHEMA_VERSION,
        "artifact_set_name": "reference_embeddings",
        "artifact_set_version": artifact_fingerprint,
        "status": "complete",
        "run_id": run_id,
        "job_name": job_name,
        "stage": stage,
        "registry_version": registry_version,
        "git_sha": _provenance_git_sha(git_sha),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "effective_configuration": {
            "embedding_dimension": int(frame["embedding_dimension"][0]),
            "input_contract_version": str(frame["input_contract_version"][0]),
            "model_id": str(frame["model_id"][0]),
            "model_revision": str(frame["model_revision"][0]),
            "preprocessing_version": str(frame["preprocessing_version"][0]),
        },
        "files": [dict(record) for record in file_records],
        "dependency_fingerprints": {
            "model_fingerprint": str(frame["model_fingerprint"][0]),
            "model_input_fingerprint": str(frame["model_input_fingerprint"][0]),
            "readiness_sha256": str(frame["readiness_sha256"][0]),
            "reference_bank_fingerprint": str(frame["reference_bank_fingerprint"][0]),
            "support_manifest_fingerprint": str(
                frame["support_manifest_fingerprint"][0]
            ),
        },
        "source_snapshot_versions": sorted(
            {str(value) for value in frame["source_snapshot_version"].to_list()}
        ),
        "qa": {"status": "passed", "fatal_count": 0, "warning_count": 0},
        "metrics": {
            "row_count": frame.height,
            "retry_count": None,
            "gpu_memory_bytes": "not_instrumented",
            "peak_rss_bytes": "not_instrumented",
        },
    }


def _validate_reference_embeddings_cloud_manifest(
    manifest: Mapping[str, object],
    *,
    storage: CloudStorage,
    frame: pl.DataFrame,
    manifest_uri: str,
    embeddings_uri: str,
    report_uri: str,
    summary_uri: str,
    artifact_fingerprint: str,
    registry_version: str,
    run_id: str,
    job_name: str,
    stage: str,
    git_sha: str,
) -> None:
    required = {
        "schema_version",
        "artifact_set_name",
        "artifact_set_version",
        "status",
        "run_id",
        "job_name",
        "stage",
        "registry_version",
        "git_sha",
        "started_at",
        "ended_at",
        "effective_configuration",
        "files",
        "dependency_fingerprints",
        "source_snapshot_versions",
        "qa",
        "metrics",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise ValueError("reference embeddings cloud manifest schema mismatch")
    expected_scalars = {
        "schema_version": REFERENCE_EMBEDDINGS_MANIFEST_SCHEMA_VERSION,
        "artifact_set_name": "reference_embeddings",
        "artifact_set_version": artifact_fingerprint,
        "status": "complete",
        "run_id": run_id,
        "job_name": job_name,
        "stage": stage,
        "registry_version": registry_version,
        "git_sha": _provenance_git_sha(git_sha),
    }
    mismatches = [
        field
        for field, expected in expected_scalars.items()
        if manifest.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            "reference embeddings cloud manifest identity mismatch: "
            + ", ".join(sorted(mismatches))
        )
    manifest_git_sha = _provenance_git_sha(manifest.get("git_sha"))
    started_at = _utc_datetime(
        datetime.fromisoformat(str(manifest["started_at"])),
        field="reference embeddings manifest started_at",
    )
    ended_at = _utc_datetime(
        datetime.fromisoformat(str(manifest["ended_at"])),
        field="reference embeddings manifest ended_at",
    )
    if ended_at < started_at:
        raise ValueError("reference embeddings cloud manifest time range is invalid")
    expected_uris = {
        REFERENCE_EMBEDDINGS_FILE: embeddings_uri,
        REFERENCE_EMBEDDINGS_REPORT_FILE: report_uri,
        REFERENCE_EMBEDDINGS_SUMMARY_FILE: summary_uri,
    }
    files = manifest.get("files")
    if (
        not isinstance(files, list)
        or len(files) != len(expected_uris)
        or {str(item.get("name")) for item in files if isinstance(item, Mapping)}
        != set(expected_uris)
    ):
        raise ValueError("reference embeddings cloud manifest file set mismatch")
    for name, expected_uri in expected_uris.items():
        record = _manifest_file_record(manifest, name)
        if record.get("uri") != expected_uri or not storage.exists(expected_uri):
            raise ValueError("reference embeddings cloud manifest file URI mismatch")
        if storage.file_size(expected_uri) != _non_negative_integer(
            record.get("byte_count"),
            field="manifest file byte_count",
        ):
            raise ValueError("reference embeddings cloud manifest file size mismatch")
        if storage.file_sha256(expected_uri) != _sha256(
            record.get("sha256"),
            field="manifest file SHA-256",
        ):
            raise ValueError("reference embeddings cloud manifest checksum mismatch")
    embedding_record = _manifest_file_record(
        manifest,
        REFERENCE_EMBEDDINGS_FILE,
    )
    if (
        embedding_record.get("semantic_fingerprint") != artifact_fingerprint
        or embedding_record.get("physical_schema_version")
        != REFERENCE_EMBEDDINGS_SCHEMA_VERSION
        or embedding_record.get("row_count") != frame.height
    ):
        raise ValueError("reference embeddings cloud manifest artifact mismatch")
    loaded = storage.read_parquet(embeddings_uri)
    validate_reference_embeddings(loaded)
    if reference_embeddings_artifact_fingerprint(loaded) != artifact_fingerprint:
        raise ValueError("reference embeddings cloud artifact fingerprint mismatch")
    if not storage.exists(manifest_uri):
        raise ValueError("reference embeddings cloud manifest is not durable")
    report = storage.read_json(report_uri)
    _validate_existing_reference_embedding_report(
        report,
        frame=frame,
        artifact_uri=embeddings_uri,
        artifact_byte_count=storage.file_size(embeddings_uri),
        artifact_fingerprint=artifact_fingerprint,
        artifact_sha256=storage.file_sha256(embeddings_uri),
        run_id=run_id,
        git_sha=manifest_git_sha,
        command="bioclip.publish_reference_embeddings_to_cloud",
        worker_id=_required_text(
            report.get("worker_id"),
            field="report worker_id",
        ),
    )
    if storage.read_text(summary_uri) != _publication_markdown(report):
        raise ValueError("reference embeddings cloud summary/report mismatch")
    if (
        report.get("git_sha") != manifest_git_sha
        or report.get("started_at") != manifest.get("started_at")
        or report.get("ended_at") != manifest.get("ended_at")
    ):
        raise ValueError("reference embeddings cloud manifest report binding mismatch")
    expected_manifest = _reference_embeddings_cloud_manifest(
        frame,
        storage=storage,
        embeddings_uri=embeddings_uri,
        report_uri=report_uri,
        summary_uri=summary_uri,
        artifact_fingerprint=artifact_fingerprint,
        registry_version=registry_version,
        run_id=run_id,
        job_name=job_name,
        stage=stage,
        git_sha=manifest_git_sha,
        started_at=started_at,
        ended_at=ended_at,
    )
    authoritative_fields = (
        "effective_configuration",
        "files",
        "dependency_fingerprints",
        "source_snapshot_versions",
        "qa",
        "metrics",
    )
    mismatches = [
        field
        for field in authoritative_fields
        if manifest.get(field) != expected_manifest[field]
    ]
    if mismatches:
        raise ValueError(
            "reference embeddings cloud manifest authoritative metadata mismatch: "
            + ", ".join(mismatches)
        )


def _manifest_file_record(
    manifest: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("reference embeddings cloud manifest files are invalid")
    matches = [
        item for item in files if isinstance(item, Mapping) and item.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError("reference embeddings cloud manifest file set mismatch")
    record = matches[0]
    if set(record) != {
        "name",
        "uri",
        "byte_count",
        "row_count",
        "physical_schema_version",
        "sha256",
        "semantic_fingerprint",
        "primary_key",
        "sort_order",
    }:
        raise ValueError("reference embeddings cloud manifest file schema mismatch")
    return record


def _validate_existing_reference_embedding_report(
    report: Mapping[str, object],
    *,
    frame: pl.DataFrame,
    artifact_uri: str,
    artifact_byte_count: int,
    artifact_fingerprint: str,
    artifact_sha256: str,
    run_id: str,
    git_sha: str,
    command: str,
    worker_id: str | None,
) -> None:
    expected_git_sha = _provenance_git_sha(git_sha)
    started_at = _utc_datetime(
        datetime.fromisoformat(str(report.get("started_at"))),
        field="reference embedding report started_at",
    )
    ended_at = _utc_datetime(
        datetime.fromisoformat(str(report.get("ended_at"))),
        field="reference embedding report ended_at",
    )
    expected = _publication_report(
        frame,
        artifact_uri=artifact_uri,
        artifact_byte_count=artifact_byte_count,
        artifact_sha256=artifact_sha256,
        run_id=run_id,
        git_sha=expected_git_sha,
        pid=_positive_integer(report.get("pid"), field="report pid"),
        started_at=started_at,
        ended_at=ended_at,
        command=command,
        worker_id=worker_id,
    )
    if report != expected or expected["artifact_fingerprint"] != artifact_fingerprint:
        raise ValueError("reference embeddings cloud report identity mismatch")


def _write_cloud_json_immutable(
    storage: CloudStorage,
    uri: str,
    value: Mapping[str, object],
) -> None:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    _write_cloud_text_immutable(
        storage,
        uri,
        encoded,
        content_type="application/json",
    )


def _write_cloud_text_immutable(
    storage: CloudStorage,
    uri: str,
    value: str,
    *,
    content_type: str,
) -> None:
    if storage.exists(uri):
        if storage.read_text(uri) != value:
            raise FileExistsError(f"immutable cloud artifact already exists: {uri}")
        return
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(mode="wb", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(value.encode("utf-8"))
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            storage.write_file(
                uri,
                temporary_path,
                content_type=content_type,
                overwrite=False,
            )
        except FileExistsError:
            if storage.read_text(uri) != value:
                raise
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _checkpoint_build_fingerprint(
    prepared: Sequence[_PreparedVisualInput],
    *,
    readiness_permit: ReferenceBankReadinessPermit,
    model_id: str,
    model_revision: str,
    preprocessing_contract: TargetPreprocessingContract,
) -> str:
    return _json_fingerprint(
        {
            "checkpoint_schema_version": (
                REFERENCE_EMBEDDINGS_CHECKPOINT_SCHEMA_VERSION
            ),
            "schema_version": REFERENCE_EMBEDDINGS_SCHEMA_VERSION,
            "reference_bank_fingerprint": readiness_permit.bank_fingerprint,
            "support_manifest_fingerprint": (
                readiness_permit.support_manifest_fingerprint
            ),
            "model_input_fingerprint": readiness_permit.model_input_fingerprint,
            "input_contract_version": readiness_permit.input_contract_version,
            "model_id": model_id,
            "model_revision": model_revision,
            "model_weights_sha256": readiness_permit.checkpoint_sha256,
            "open_clip_version": readiness_permit.open_clip_version,
            "open_clip_config_sha256": (readiness_permit.open_clip_config_sha256),
            "preprocessing_attestation_fingerprint": (
                readiness_permit.preprocessing_attestation_fingerprint
            ),
            "preprocessing_version": preprocessing_contract.version,
            "preprocessing_fingerprint": preprocessing_contract.fingerprint,
            "visual_inputs": [
                {
                    "reference_media_id": item.visual_input.reference_media_id,
                    "support_row_fingerprint": item.support_row[
                        "support_row_fingerprint"
                    ],
                    "source_image_sha256": item.support_row["image_sha256"],
                    "visual_input_id": item.visual_input.visual_input_id,
                    "visual_input_kind": item.visual_input.visual_input_kind,
                    "raw_image_content_hash": (
                        item.visual_input.raw_image_content_hash
                    ),
                    "image_content_hash": item.visual_input.image_content_hash,
                    "transformation_version": (
                        item.visual_input.transformation_version
                    ),
                    "transformation_policy_fingerprint": (
                        item.visual_input.transformation_policy_fingerprint
                    ),
                    "transformation_fingerprint": (
                        item.visual_input.transformation_fingerprint
                    ),
                }
                for item in prepared
            ],
        }
    )


def _acquire_embedding_checkpoint_lock(
    checkpoint_root: Path | None,
    *,
    build_fingerprint: str,
) -> BinaryIO | None:
    if checkpoint_root is None:
        return None
    checkpoint_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint_root.parent / (
        f".{checkpoint_root.name}{REFERENCE_EMBEDDINGS_CHECKPOINT_LOCK_SUFFIX}"
    )
    stream = lock_path.open("a+b")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        if isinstance(exc, BlockingIOError):
            raise ReferenceEmbeddingCheckpointBusyError(
                f"reference embedding checkpoint writer is busy: {checkpoint_root}"
            ) from exc
        raise
    owner = {
        "build_fingerprint": build_fingerprint,
        "pid": os.getpid(),
        "started_at": datetime.now(UTC).isoformat(),
    }
    stream.seek(0)
    stream.truncate()
    stream.write(
        (json.dumps(owner, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "ascii"
        )
    )
    stream.flush()
    os.fsync(stream.fileno())
    return stream


def _release_embedding_checkpoint_lock(stream: BinaryIO | None) -> None:
    if stream is None:
        return
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def _load_embedding_checkpoint(
    checkpoint_root: Path | None,
    *,
    expected_build_fingerprint: str,
    requested_created_at: datetime | None,
    resume: bool,
) -> _LoadedEmbeddingCheckpoint:
    if checkpoint_root is None:
        return _LoadedEmbeddingCheckpoint()
    state_path = checkpoint_root / REFERENCE_EMBEDDINGS_CHECKPOINT_STATE_FILE
    if not state_path.is_file():
        if not resume and checkpoint_root.exists() and any(checkpoint_root.iterdir()):
            raise FileExistsError(
                f"reference embedding checkpoint already exists: {checkpoint_root}"
            )
        unexpected = (
            [
                path
                for path in checkpoint_root.iterdir()
                if path.name != REFERENCE_EMBEDDINGS_CHECKPOINT_PARTS_DIR
                and not (
                    path.name.startswith(
                        f".{REFERENCE_EMBEDDINGS_CHECKPOINT_STATE_FILE}."
                    )
                    and path.name.endswith(".tmp")
                )
            ]
            if checkpoint_root.exists()
            else []
        )
        if unexpected:
            raise ValueError(
                "reference embedding checkpoint is partial: unexpected files "
                "exist without state"
            )
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        orphan_count = _remove_unreferenced_checkpoint_parts(
            checkpoint_root,
            referenced_filenames=frozenset(),
        )
        if orphan_count:
            _log_event(
                "reference_embedding_checkpoint_orphans_removed",
                checkpoint_dir=str(checkpoint_root),
                orphan_count=orphan_count,
            )
        return _LoadedEmbeddingCheckpoint()
    if not resume:
        raise FileExistsError(
            f"reference embedding checkpoint already exists: {state_path}"
        )
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("reference embedding checkpoint state is unreadable") from exc
    if not isinstance(state, dict) or set(state) != {
        "schema_version",
        "build_fingerprint",
        "embedding_created_at",
        "embedding_dimension",
        "model_weights_sha256",
        "preprocessing_attestation_fingerprint",
        "generation",
        "row_count",
        "parts",
    }:
        raise ValueError("reference embedding checkpoint state schema mismatch")
    if state["schema_version"] != REFERENCE_EMBEDDINGS_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported reference embedding checkpoint schema version")
    if state["build_fingerprint"] != expected_build_fingerprint:
        raise ValueError("reference embedding checkpoint build fingerprint mismatch")
    generation = _non_negative_integer(
        state["generation"],
        field="checkpoint generation",
    )
    if generation == 0:
        raise ValueError("reference embedding checkpoint generation must be positive")
    try:
        created_at = _utc_datetime(
            datetime.fromisoformat(str(state["embedding_created_at"])),
            field="checkpoint embedding_created_at",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "reference embedding checkpoint creation time is invalid"
        ) from exc
    if requested_created_at is not None and requested_created_at != created_at:
        _log_event(
            "reference_embedding_checkpoint_creation_time_reused",
            checkpoint_dir=str(checkpoint_root),
            persisted_embedding_created_at=created_at.isoformat(),
            requested_embedding_created_at=requested_created_at.isoformat(),
        )
    dimension = _positive_dimension(state["embedding_dimension"])
    weights_sha256 = _sha256(
        state["model_weights_sha256"],
        field="checkpoint model_weights_sha256",
    )
    attestation_fingerprint = _sha256(
        state["preprocessing_attestation_fingerprint"],
        field="checkpoint preprocessing_attestation_fingerprint",
    )
    parts = state["parts"]
    if not isinstance(parts, list) or not parts:
        raise ValueError("reference embedding checkpoint has no durable parts")
    frames: list[pl.DataFrame] = []
    seen_parts: set[str] = set()
    for part in parts:
        if not isinstance(part, dict) or set(part) != {
            "artifact_fingerprint",
            "file",
            "row_count",
            "sha256",
        }:
            raise ValueError("reference embedding checkpoint part schema mismatch")
        filename = _required_text(part["file"], field="checkpoint part file")
        if Path(filename).name != filename or filename in seen_parts:
            raise ValueError("reference embedding checkpoint part filename is invalid")
        seen_parts.add(filename)
        part_path = (
            checkpoint_root / REFERENCE_EMBEDDINGS_CHECKPOINT_PARTS_DIR / filename
        )
        if not part_path.is_file():
            raise ValueError("reference embedding checkpoint part is missing")
        if _file_sha256(part_path) != _sha256(
            part["sha256"], field="checkpoint part SHA-256"
        ):
            raise ValueError("reference embedding checkpoint part SHA-256 mismatch")
        part_frame = pl.read_parquet(part_path)
        validate_reference_embeddings(
            part_frame,
            expected_model_weights_sha256=weights_sha256,
            expected_preprocessing_attestation_fingerprint=(attestation_fingerprint),
            require_raw_full_image=False,
        )
        if part_frame.height != int(part["row_count"]):
            raise ValueError("reference embedding checkpoint part row count mismatch")
        if (
            reference_embeddings_artifact_fingerprint(
                part_frame,
                require_raw_full_image=False,
            )
            != part["artifact_fingerprint"]
        ):
            raise ValueError("reference embedding checkpoint part fingerprint mismatch")
        frames.append(part_frame)
    orphan_count = _remove_unreferenced_checkpoint_parts(
        checkpoint_root,
        referenced_filenames=frozenset(seen_parts),
    )
    if orphan_count:
        _log_event(
            "reference_embedding_checkpoint_orphans_removed",
            checkpoint_dir=str(checkpoint_root),
            orphan_count=orphan_count,
        )
    frame = _sort_embedding_frame(pl.concat(frames, rechunk=True))
    # Checkpoint parts can span relocations of semantically identical support and
    # readiness objects. Each part is validated above; the builder rebinds every
    # row to the current permit and manifest before any additional model work.
    if frame.height != int(state["row_count"]):
        raise ValueError("reference embedding checkpoint row count mismatch")
    if int(frame["embedding_dimension"][0]) != dimension:
        raise ValueError("reference embedding checkpoint dimension mismatch")
    timestamps = set(frame["embedding_created_at"].to_list())
    if timestamps != {created_at}:
        raise ValueError("reference embedding checkpoint creation time drift")
    attestation = _preprocessing_attestation_from_row(frame.row(0, named=True))
    return _LoadedEmbeddingCheckpoint(
        frame=frame,
        embedding_created_at=created_at,
        embedding_dimension=dimension,
        model_weights_sha256=weights_sha256,
        preprocessing_attestation=attestation,
        generation=generation,
        parts_fingerprint=_json_fingerprint(parts),
    )


def _write_embedding_checkpoint_batch(
    checkpoint_root: Path,
    *,
    batch_frame: pl.DataFrame,
    completed_row_count: int,
    build_fingerprint: str,
    embedding_created_at: datetime,
    embedding_dimension: int,
    model_weights_sha256: str,
    preprocessing_attestation: _PreprocessingAttestation,
    expected_generation: int,
    expected_parts_fingerprint: str,
) -> tuple[int, str]:
    parts_dir = checkpoint_root / REFERENCE_EMBEDDINGS_CHECKPOINT_PARTS_DIR
    parts_dir.mkdir(parents=True, exist_ok=True)
    validate_reference_embeddings(
        batch_frame,
        require_raw_full_image=False,
    )
    filename = f"batch-{uuid4().hex}.parquet"
    part_path = write_parquet(
        batch_frame,
        parts_dir / filename,
        overwrite=False,
    )
    _fsync_file(part_path)
    _fsync_directory(parts_dir)
    part = {
        "artifact_fingerprint": reference_embeddings_artifact_fingerprint(
            batch_frame,
            require_raw_full_image=False,
        ),
        "file": filename,
        "row_count": batch_frame.height,
        "sha256": _file_sha256(part_path),
    }
    state_path = checkpoint_root / REFERENCE_EMBEDDINGS_CHECKPOINT_STATE_FILE
    existing_parts: list[object] = []
    existing_row_count = 0
    if state_path.is_file():
        try:
            existing = json.loads(state_path.read_text(encoding="utf-8"))
            existing_parts = list(existing["parts"])
            existing_generation = _non_negative_integer(
                existing["generation"],
                field="checkpoint generation",
            )
            existing_row_count = _non_negative_integer(
                existing["row_count"],
                field="checkpoint row count",
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "reference embedding checkpoint state became unreadable"
            ) from exc
        if existing.get("build_fingerprint") != build_fingerprint:
            raise ValueError("reference embedding checkpoint writer build mismatch")
        if existing_generation != expected_generation:
            raise ValueError("reference embedding checkpoint generation changed")
        if _json_fingerprint(existing_parts) != expected_parts_fingerprint:
            raise ValueError("reference embedding checkpoint parts changed")
    elif expected_generation != 0 or expected_parts_fingerprint != _json_fingerprint(
        []
    ):
        raise ValueError("reference embedding checkpoint state disappeared")
    if completed_row_count != existing_row_count + batch_frame.height:
        raise ValueError("reference embedding checkpoint row count transition mismatch")
    next_parts = [*existing_parts, part]
    part_names = [str(item["file"]) for item in next_parts if isinstance(item, dict)]
    if len(part_names) != len(next_parts) or len(part_names) != len(set(part_names)):
        raise ValueError("reference embedding checkpoint parts are invalid")
    persisted_row_count = sum(
        _non_negative_integer(item["row_count"], field="checkpoint part row count")
        for item in next_parts
        if isinstance(item, dict)
    )
    if persisted_row_count != completed_row_count:
        raise ValueError("reference embedding checkpoint part rows do not match state")
    next_generation = expected_generation + 1
    state = {
        "schema_version": REFERENCE_EMBEDDINGS_CHECKPOINT_SCHEMA_VERSION,
        "build_fingerprint": build_fingerprint,
        "embedding_created_at": embedding_created_at.isoformat(),
        "embedding_dimension": embedding_dimension,
        "model_weights_sha256": model_weights_sha256,
        "preprocessing_attestation_fingerprint": (
            preprocessing_attestation.preprocessing_fingerprint
        ),
        "generation": next_generation,
        "row_count": completed_row_count,
        "parts": next_parts,
    }
    _write_json_atomically(state_path, state)
    _log_event(
        "reference_embedding_checkpoint_written",
        checkpoint_state=str(state_path),
        checkpoint_part=str(part_path),
        checkpoint_rows=completed_row_count,
        checkpoint_bytes=part_path.stat().st_size,
    )
    return next_generation, _json_fingerprint(next_parts)


def _remove_unreferenced_checkpoint_parts(
    checkpoint_root: Path,
    *,
    referenced_filenames: frozenset[str],
) -> int:
    parts_dir = checkpoint_root / REFERENCE_EMBEDDINGS_CHECKPOINT_PARTS_DIR
    if not parts_dir.exists():
        return 0
    if not parts_dir.is_dir():
        raise ValueError("reference embedding checkpoint parts path is not a directory")
    removed = 0
    unexpected: list[str] = []
    for path in parts_dir.iterdir():
        if path.name in referenced_filenames:
            continue
        if (
            path.is_file()
            and path.name.startswith("batch-")
            and path.suffix == ".parquet"
        ):
            path.unlink()
            removed += 1
            continue
        unexpected.append(path.name)
    if unexpected:
        raise ValueError(
            "reference embedding checkpoint contains unexpected part files: "
            + ", ".join(sorted(unexpected)[:10])
        )
    if removed:
        _fsync_directory(parts_dir)
    return removed


def _validated_support_manifest(
    frame: pl.DataFrame,
    *,
    readiness_permit: ReferenceBankReadinessPermit,
) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("support_manifest must be a Polars DataFrame")
    if not isinstance(readiness_permit, ReferenceBankReadinessPermit):
        raise TypeError("readiness_permit must be a ReferenceBankReadinessPermit")
    if readiness_permit.status not in PERMITTING_READINESS_STATUSES:
        raise ValueError(
            "reference readiness permit does not authorize support embeddings"
        )
    for field in (
        "readiness_sha256",
        "bank_fingerprint",
        "support_manifest_fingerprint",
        "model_input_fingerprint",
        "checkpoint_sha256",
        "open_clip_config_sha256",
        "preprocessing_contract_fingerprint",
        "preprocessing_attestation_fingerprint",
    ):
        _sha256(getattr(readiness_permit, field), field=f"readiness {field}")
    _validate_absolute_uri(
        readiness_permit.checkpoint_uri,
        field="readiness checkpoint_uri",
    )
    for field in (
        "model_name",
        "model_version",
        "model_revision",
        "open_clip_version",
        "preprocessing_version",
        "input_contract_version",
    ):
        _required_text(getattr(readiness_permit, field), field=f"readiness {field}")
    if frame.columns != list(reference_support_manifest_schema()):
        raise ValueError("reference support manifest physical schema mismatch")
    canonical = frame.sort(list(_SUPPORT_MANIFEST_SORT))
    validate_reference_support_manifest(canonical)
    manifest_fingerprint = reference_support_manifest_fingerprint(canonical)
    if manifest_fingerprint != readiness_permit.support_manifest_fingerprint:
        raise ValueError("reference readiness support manifest fingerprint mismatch")
    registry_versions = set(canonical["registry_version"].to_list())
    if registry_versions != {readiness_permit.registry_version}:
        raise ValueError("reference readiness registry version mismatch")
    bank_versions = set(canonical["reference_bank_version"].to_list())
    if bank_versions != {readiness_permit.reference_bank_version}:
        raise ValueError("reference readiness bank version mismatch")
    bank_fingerprints = set(canonical["reference_bank_fingerprint"].to_list())
    if bank_fingerprints != {readiness_permit.bank_fingerprint}:
        raise ValueError("reference readiness bank fingerprint mismatch")
    leakage = reference_support_split_leakage(canonical)
    if leakage:
        first = leakage[0]
        raise ValueError(
            "reference support manifest leaks provenance across splits: "
            f"{first['group_type']}={first['group_value']}"
        )
    return canonical


def _prepare_visual_inputs(
    eligible: pl.DataFrame,
    visual_inputs: Sequence[ReferenceVisualInput],
) -> list[_PreparedVisualInput]:
    support_by_media = {
        str(row["reference_media_id"]): row for row in eligible.iter_rows(named=True)
    }
    prepared: list[_PreparedVisualInput] = []
    checkpoint_keys: set[tuple[str, str]] = set()
    for visual_input in visual_inputs:
        if not isinstance(visual_input, ReferenceVisualInput):
            raise TypeError("visual_inputs must contain ReferenceVisualInput values")
        support_row = support_by_media.get(visual_input.reference_media_id)
        if support_row is None:
            raise ValueError(
                "reference visual input identifies unknown or ineligible media: "
                f"{visual_input.reference_media_id}"
            )
        item = _PreparedVisualInput(
            visual_input=visual_input,
            support_row=support_row,
        )
        checkpoint_key = _prepared_checkpoint_key(item)
        if checkpoint_key in checkpoint_keys:
            raise ValueError(
                "duplicate reference visual input checkpoint identity: "
                f"{visual_input.reference_media_id}:{visual_input.visual_input_kind}"
            )
        checkpoint_keys.add(checkpoint_key)
        prepared.append(item)
    covered = {item.visual_input.reference_media_id for item in prepared}
    missing = sorted(set(support_by_media) - covered)
    if missing:
        raise ValueError(
            "eligible reference media missing visual inputs: " + ", ".join(missing[:10])
        )
    raw_counts = Counter(
        item.visual_input.reference_media_id
        for item in prepared
        if item.visual_input.visual_input_kind == RAW_FULL_IMAGE_KIND
    )
    invalid_raw_counts = sorted(
        media_id for media_id in support_by_media if raw_counts[media_id] != 1
    )
    if invalid_raw_counts:
        raise ValueError(
            "eligible reference media require exactly one raw full-image input: "
            + ", ".join(invalid_raw_counts[:10])
        )
    split_by_hash: dict[str, str] = {}
    for item in prepared:
        split = str(item.support_row["support_split"])
        previous = split_by_hash.setdefault(
            item.visual_input.image_content_hash,
            split,
        )
        if previous != split:
            raise ValueError("reference image content crosses support splits")
    return sorted(prepared, key=_prepared_sort_key)


def _prepared_sort_key(item: _PreparedVisualInput) -> tuple[object, ...]:
    visual_input = item.visual_input
    row = item.support_row
    return (
        str(row["accepted_taxon_key"]),
        str(row["route"]),
        str(row["geo_cluster_id"]),
        str(row["support_split"]),
        visual_input.reference_media_id,
        _VISUAL_INPUT_KIND_ORDER[visual_input.visual_input_kind],
        visual_input.image_content_hash,
        visual_input.transformation_version,
        visual_input.transformation_fingerprint,
        visual_input.visual_input_id,
    )


def _expected_visual_input_id(visual_input: ReferenceVisualInput) -> str:
    return _expected_visual_input_id_from_values(
        visual_input_kind=visual_input.visual_input_kind,
        raw_image_content_hash=visual_input.raw_image_content_hash,
        image_content_hash=visual_input.image_content_hash,
        transformation_fingerprint=visual_input.transformation_fingerprint,
    )


def _expected_visual_input_id_from_values(
    *,
    visual_input_kind: str,
    raw_image_content_hash: str,
    image_content_hash: str,
    transformation_fingerprint: str,
) -> str:
    payload = {
        "raw_image_content_hash": raw_image_content_hash,
        "transformation_fingerprint": transformation_fingerprint,
        "visual_input_kind": visual_input_kind,
        "visual_input_version": FULL_FRAME_VISUAL_INPUT_VERSION,
    }
    if visual_input_kind != RAW_FULL_IMAGE_KIND:
        payload["visual_content_hash"] = image_content_hash
    return canonical_semantic_fingerprint(payload)


def _prepared_checkpoint_key(item: _PreparedVisualInput) -> tuple[str, str]:
    return (
        str(item.support_row["support_row_fingerprint"]),
        item.visual_input.visual_input_id,
    )


def _prepared_embedding_cache_key(
    item: _PreparedVisualInput,
    *,
    input_contract_version: str,
    model_id: str,
    model_revision: str,
    preprocessing_version: str,
    model_input_fingerprint: str,
) -> tuple[str, str, str, str, str, str]:
    return (
        _sha256(
            item.visual_input.image_content_hash,
            field="image_content_hash",
        ),
        _required_text(
            input_contract_version,
            field="input_contract_version",
        ),
        _required_text(model_id, field="model_id"),
        _required_text(model_revision, field="model_revision"),
        _required_text(preprocessing_version, field="preprocessing_version"),
        _sha256(model_input_fingerprint, field="model_input_fingerprint"),
    )


def _embedding_cache_key_from_row(
    row: Mapping[str, object],
) -> tuple[str, str, str, str, str, str]:
    return (
        _sha256(row["image_content_hash"], field="image_content_hash"),
        _required_text(
            row["input_contract_version"],
            field="input_contract_version",
        ),
        _required_text(row["model_id"], field="model_id"),
        _required_text(row["model_revision"], field="model_revision"),
        _required_text(
            row["preprocessing_version"],
            field="preprocessing_version",
        ),
        _sha256(
            row["model_input_fingerprint"],
            field="model_input_fingerprint",
        ),
    )


def _load_reference_embedding_vector_cache(
    cache: pl.DataFrame | str | Path | None,
    *,
    readiness_permit: ReferenceBankReadinessPermit,
    model_id: str,
    model_revision: str,
    preprocessing_contract: TargetPreprocessingContract,
    preprocessing_attestation: _PreprocessingAttestation,
) -> tuple[
    dict[tuple[str, str, str, str, str, str], tuple[tuple[float, ...], float]],
    int | None,
]:
    if cache is None:
        return {}, None
    validation_kwargs = {
        "expected_model_id": model_id,
        "expected_model_revision": model_revision,
        "expected_model_weights_sha256": readiness_permit.checkpoint_sha256,
        "expected_preprocessing_version": preprocessing_contract.version,
        "expected_preprocessing_fingerprint": preprocessing_contract.fingerprint,
        "expected_preprocessing_attestation_fingerprint": (
            preprocessing_attestation.preprocessing_fingerprint
        ),
        "expected_model_input_fingerprint": (readiness_permit.model_input_fingerprint),
        "expected_input_contract_version": (readiness_permit.input_contract_version),
    }
    if isinstance(cache, pl.DataFrame):
        frame = cache
        validate_reference_embeddings(frame, **validation_kwargs)
    elif isinstance(cache, (str, Path)):
        frame = _read_reference_embeddings_parquet(cache, **validation_kwargs)
    else:
        raise TypeError(
            "reference embedding cache must be a Polars DataFrame or Parquet path"
        )
    vectors: dict[
        tuple[str, str, str, str, str, str], tuple[tuple[float, ...], float]
    ] = {}
    _merge_reference_embedding_vectors_into_cache(vectors, frame)
    return vectors, _positive_dimension(frame["embedding_dimension"][0])


def _merge_reference_embedding_vectors_into_cache(
    cache: dict[tuple[str, str, str, str, str, str], tuple[tuple[float, ...], float]],
    frame: pl.DataFrame,
) -> None:
    for row in frame.iter_rows(named=True):
        key = _embedding_cache_key_from_row(row)
        vector = tuple(float(value) for value in row["embedding"])
        norm = float(row["embedding_norm"])
        value = (vector, norm)
        previous = cache.setdefault(key, value)
        if previous != value:
            raise ValueError(
                "reference embedding cache has conflicting vectors for one "
                "content and model identity"
            )


def _rebind_resumed_checkpoint_provenance(
    resumed_frame: pl.DataFrame | None,
    prepared: Sequence[_PreparedVisualInput],
    *,
    readiness_permit: ReferenceBankReadinessPermit,
    model_id: str,
    model_revision: str,
    preprocessing_contract: TargetPreprocessingContract,
) -> pl.DataFrame | None:
    if resumed_frame is None:
        return None
    prepared_by_key = {_prepared_checkpoint_key(item): item for item in prepared}
    rebound_rows: list[dict[str, object]] = []
    relocatable_fields = {
        "model_checkpoint_uri",
        "readiness_sha256",
        "review_decision_ids",
        "source_object_uri",
        "source_object_fingerprint",
    }
    for row in resumed_frame.iter_rows(named=True):
        key = (str(row["support_row_fingerprint"]), str(row["visual_input_id"]))
        item = prepared_by_key.get(key)
        if item is None:
            raise ValueError(
                "reference embedding checkpoint contains unknown completed inputs"
            )
        expected = _embedding_row(
            item,
            vector=tuple(float(value) for value in row["embedding"]),
            embedding_norm=float(row["embedding_norm"]),
            embedding_dimension=int(row["embedding_dimension"]),
            model_id=model_id,
            model_revision=model_revision,
            model_weights_sha256=readiness_permit.checkpoint_sha256,
            preprocessing_contract=preprocessing_contract,
            preprocessing_attestation=_preprocessing_attestation_from_row(row),
            readiness_permit=readiness_permit,
            embedding_created_at=_utc_datetime(
                row["embedding_created_at"],
                field="checkpoint embedding_created_at",
            ),
        )
        expected["embedding_fingerprint"] = _embedding_row_fingerprint(expected)
        mismatches = sorted(
            field
            for field in expected
            if field not in relocatable_fields and row.get(field) != expected[field]
        )
        if mismatches:
            raise ValueError(
                "reference embedding checkpoint provenance mismatch: "
                + ", ".join(mismatches)
            )
        rebound_rows.append(expected)
    rebound = _sort_embedding_frame(
        pl.DataFrame(
            rebound_rows,
            schema=reference_embeddings_schema(
                int(resumed_frame["embedding_dimension"][0])
            ),
            orient="row",
            strict=True,
        )
    )
    validate_reference_embeddings(rebound, require_raw_full_image=False)
    return rebound


def _validate_visual_input_files(
    prepared: Sequence[_PreparedVisualInput],
) -> None:
    source_metrics_by_path: dict[Path, tuple[str, str]] = {}
    visual_hash_by_path: dict[Path, str] = {}
    for item in prepared:
        _validate_source_path_binding(item)
        source_path = item.visual_input.source_image_path.resolve(strict=True)
        source_metrics = source_metrics_by_path.get(source_path)
        if source_metrics is None:
            source_metrics = (
                _file_sha256(source_path),
                decoded_image_file_content_hash(source_path),
            )
            source_metrics_by_path[source_path] = source_metrics
        source_sha256, raw_hash = source_metrics
        if source_sha256 != item.support_row["image_sha256"]:
            raise ValueError(
                "reference visual input source object SHA-256 mismatch: "
                f"{item.visual_input.reference_media_id}"
            )
        if raw_hash != item.visual_input.raw_image_content_hash:
            raise ValueError(
                "reference visual input raw content hash mismatch: "
                f"{item.visual_input.reference_media_id}"
            )
        image_path = item.visual_input.image_path.resolve(strict=True)
        actual_hash = visual_hash_by_path.get(image_path)
        if actual_hash is None:
            actual_hash = (
                raw_hash
                if image_path == source_path
                else decoded_image_file_content_hash(image_path)
            )
            visual_hash_by_path[image_path] = actual_hash
        if actual_hash != item.visual_input.image_content_hash:
            raise ValueError(
                "reference visual input decoded content hash mismatch: "
                f"{item.visual_input.reference_media_id}"
            )
        if item.visual_input.visual_input_id != _expected_visual_input_id(
            item.visual_input
        ):
            raise ValueError(
                "reference visual input identity fingerprint mismatch: "
                f"{item.visual_input.reference_media_id}"
            )


def _validate_source_path_binding(item: _PreparedVisualInput) -> None:
    source_path = item.visual_input.source_image_path
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_uri = _validate_absolute_uri(
        item.support_row["source_object_uri"],
        field="source_object_uri",
    )
    parsed = urlsplit(source_uri)
    if parsed.scheme.casefold() != "file":
        return
    if parsed.netloc not in {"", "localhost"}:
        raise ValueError("reference source object file URI host is unsupported")
    reviewed_path = Path(unquote(parsed.path)).resolve(strict=True)
    if source_path.resolve(strict=True) != reviewed_path:
        raise ValueError(
            "reference visual input source path does not match reviewed object URI: "
            f"{item.visual_input.reference_media_id}"
        )


def _validated_scorer_preprocessing_attestation(
    scorer: ReferenceImageEmbeddingScorer,
    *,
    contract: TargetPreprocessingContract,
) -> _PreprocessingAttestation:
    open_clip_version = _required_scorer_text(scorer, "open_clip_version")
    open_clip_config_sha256 = _sha256(
        getattr(scorer, "open_clip_config_sha256", None),
        field="OpenCLIP config SHA-256",
    )
    preprocessing_version = _required_scorer_text(
        scorer,
        "preprocessing_version",
    )
    config_json, config = _canonical_preprocessing_config(
        getattr(scorer, "preprocessing_config", None)
    )
    _validate_preprocessing_config(config, contract=contract)
    fingerprint = _sha256(
        getattr(scorer, "preprocessing_fingerprint", None),
        field="preprocessing attestation fingerprint",
    )
    expected = _preprocessing_attestation_fingerprint(
        open_clip_version=open_clip_version,
        open_clip_config_sha256=open_clip_config_sha256,
        preprocessing_version=preprocessing_version,
        preprocessing_config=config,
    )
    if fingerprint != expected:
        raise ValueError("reference image scorer preprocessing attestation mismatch")
    return _PreprocessingAttestation(
        open_clip_version=open_clip_version,
        open_clip_config_sha256=open_clip_config_sha256,
        preprocessing_version=preprocessing_version,
        preprocessing_config_json=config_json,
        preprocessing_fingerprint=fingerprint,
    )


def _validated_scorer_runtime_identity(
    scorer: ReferenceImageEmbeddingScorer,
    *,
    contract: TargetPreprocessingContract,
    readiness_permit: ReferenceBankReadinessPermit,
    model_id: str,
    model_revision: str,
) -> tuple[str, _PreprocessingAttestation]:
    if _required_scorer_text(scorer, "model_id") != model_id:
        raise ValueError("reference image scorer model ID changed during build")
    if _required_scorer_text(scorer, "model_revision") != model_revision:
        raise ValueError("reference image scorer model revision changed during build")
    weights_sha256 = _sha256(
        getattr(scorer, "model_weights_sha256", None),
        field="model weights SHA-256",
    )
    if weights_sha256 != readiness_permit.checkpoint_sha256:
        raise ValueError("reference readiness model weights do not match scorer")
    effective_resize_mode = _required_scorer_text(
        scorer,
        "effective_image_resize_mode",
    )
    if effective_resize_mode != TARGET_FULL_FRAME_IMAGE_RESIZE_MODE:
        raise ValueError(
            "reference image scorer effective resize mode does not match target contract"
        )
    attestation = _validated_scorer_preprocessing_attestation(
        scorer,
        contract=contract,
    )
    if attestation.open_clip_version != readiness_permit.open_clip_version:
        raise ValueError("reference readiness OpenCLIP version does not match scorer")
    if attestation.open_clip_config_sha256 != readiness_permit.open_clip_config_sha256:
        raise ValueError("reference readiness OpenCLIP config does not match scorer")
    if (
        attestation.preprocessing_fingerprint
        != readiness_permit.preprocessing_attestation_fingerprint
    ):
        raise ValueError(
            "reference readiness preprocessing attestation does not match scorer"
        )
    return weights_sha256, attestation


def _preprocessing_attestation_from_row(
    row: Mapping[str, object],
) -> _PreprocessingAttestation:
    config_json, config = _canonical_preprocessing_config(
        json.loads(
            _required_text(
                row["preprocessing_config_json"], field="preprocessing_config_json"
            )
        )
    )
    if config_json != row["preprocessing_config_json"]:
        raise ValueError("preprocessing config JSON is not canonical")
    contract = _preprocessing_contract_from_row(row, config=config)
    _validate_preprocessing_config(
        config,
        contract=contract,
    )
    attestation = _PreprocessingAttestation(
        open_clip_version=_required_text(
            row["open_clip_version"], field="open_clip_version"
        ),
        open_clip_config_sha256=_sha256(
            row["open_clip_config_sha256"], field="open_clip_config_sha256"
        ),
        preprocessing_version=_required_text(
            row["preprocessing_attestation_version"],
            field="preprocessing_attestation_version",
        ),
        preprocessing_config_json=config_json,
        preprocessing_fingerprint=_sha256(
            row["preprocessing_attestation_fingerprint"],
            field="preprocessing_attestation_fingerprint",
        ),
    )
    expected = _preprocessing_attestation_fingerprint(
        open_clip_version=attestation.open_clip_version,
        open_clip_config_sha256=attestation.open_clip_config_sha256,
        preprocessing_version=attestation.preprocessing_version,
        preprocessing_config=config,
    )
    if attestation.preprocessing_fingerprint != expected:
        raise ValueError("reference embeddings preprocessing attestation mismatch")
    return attestation


def _preprocessing_contract_from_row(
    row: Mapping[str, object],
    *,
    config: Mapping[str, object],
) -> TargetPreprocessingContract:
    dimensions = _preprocessing_dimensions(config)
    if dimensions[0] != dimensions[1]:
        raise ValueError("OpenCLIP preprocessing size must be square")
    contract = TargetPreprocessingContract(
        version=_required_text(
            row["preprocessing_version"],
            field="preprocessing_version",
        ),
        image_size_px=dimensions[0],
        normalization_mean=_preprocessing_normalization(config, field="mean"),
        normalization_std=_preprocessing_normalization(config, field="std"),
    )
    declared_fingerprint = _sha256(
        row["preprocessing_fingerprint"],
        field="preprocessing_fingerprint",
    )
    if contract.fingerprint != declared_fingerprint:
        raise ValueError("reference embeddings preprocessing contract mismatch")
    return contract


def _canonical_preprocessing_config(
    value: object,
) -> tuple[str, dict[str, object]]:
    if not isinstance(value, Mapping):
        raise ValueError("preprocessing config must be a mapping")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("preprocessing config must be canonical JSON data") from exc
    if not isinstance(decoded, dict):
        raise ValueError("preprocessing config must be a mapping")
    return encoded, decoded


def _validate_preprocessing_config(
    config: Mapping[str, object],
    *,
    contract: TargetPreprocessingContract,
) -> None:
    dimensions = _preprocessing_dimensions(config)
    if dimensions != (contract.image_size_px, contract.image_size_px):
        raise ValueError("OpenCLIP preprocessing size does not match target contract")
    if str(config.get("interpolation") or "").casefold() != contract.interpolation:
        raise ValueError(
            "OpenCLIP preprocessing interpolation does not match target contract"
        )
    if str(config.get("resize_mode") or "").casefold() != contract.resize_mode:
        raise ValueError(
            "OpenCLIP preprocessing resize mode does not match target contract"
        )
    if str(config.get("mode") or "").upper() != "RGB":
        raise ValueError(
            "OpenCLIP preprocessing colour mode does not match target contract"
        )
    fill = config.get("fill_color")
    if fill != contract.padding_fill:
        raise ValueError("OpenCLIP preprocessing fill does not match target contract")
    for field, expected in (
        ("mean", contract.normalization_mean),
        ("std", contract.normalization_std),
    ):
        if _preprocessing_normalization(config, field=field) != expected:
            raise ValueError(
                f"OpenCLIP preprocessing {field} does not match target contract"
            )


def _preprocessing_dimensions(config: Mapping[str, object]) -> tuple[int, int]:
    size = config.get("size")
    if isinstance(size, int) and not isinstance(size, bool):
        dimensions = (size, size)
    elif (
        isinstance(size, list)
        and len(size) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool) for value in size
        )
    ):
        dimensions = (size[0], size[1])
    else:
        raise ValueError("OpenCLIP preprocessing size is invalid")
    return dimensions


def _preprocessing_normalization(
    config: Mapping[str, object],
    *,
    field: str,
) -> tuple[float, float, float]:
    values = config.get(field)
    if (
        not isinstance(values, list)
        or len(values) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in values
        )
    ):
        raise ValueError(f"OpenCLIP preprocessing {field} is invalid")
    return (float(values[0]), float(values[1]), float(values[2]))


def _preprocessing_attestation_fingerprint(
    *,
    open_clip_version: str,
    open_clip_config_sha256: str,
    preprocessing_version: str,
    preprocessing_config: Mapping[str, object],
) -> str:
    return canonical_semantic_fingerprint(
        {
            "open_clip_config_sha256": open_clip_config_sha256,
            "open_clip_version": open_clip_version,
            "preprocessing_config": preprocessing_config,
            "preprocessing_version": preprocessing_version,
        }
    )


def _sort_embedding_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    order_column = "__visual_input_kind_order"
    return (
        frame.with_columns(
            pl.col("visual_input_kind")
            .replace_strict(_VISUAL_INPUT_KIND_ORDER, return_dtype=pl.UInt8)
            .alias(order_column)
        )
        .sort(
            [
                "accepted_taxon_key",
                "route",
                "geo_cluster_id",
                "support_split",
                "reference_media_id",
                order_column,
                "image_content_hash",
                "transformation_version",
                "transformation_fingerprint",
                "visual_input_id",
            ]
        )
        .drop(order_column)
    )


def _embedding_row(
    item: _PreparedVisualInput,
    *,
    vector: tuple[float, ...],
    embedding_norm: float,
    embedding_dimension: int,
    model_id: str,
    model_revision: str,
    model_weights_sha256: str,
    preprocessing_contract: TargetPreprocessingContract,
    preprocessing_attestation: _PreprocessingAttestation,
    readiness_permit: ReferenceBankReadinessPermit,
    embedding_created_at: datetime,
) -> dict[str, object]:
    support = item.support_row
    visual_input = item.visual_input
    return {
        "schema_version": REFERENCE_EMBEDDINGS_SCHEMA_VERSION,
        "registry_version": support["registry_version"],
        "reference_bank_version": support["reference_bank_version"],
        "reference_media_id": visual_input.reference_media_id,
        "reference_observation_id": support["reference_observation_id"],
        "source_snapshot_version": support["source_snapshot_version"],
        "review_decision_ids": support["review_decision_ids"],
        "duplicate_group_id": support["duplicate_group_id"],
        "readiness_sha256": readiness_permit.readiness_sha256,
        "reference_bank_fingerprint": readiness_permit.bank_fingerprint,
        "support_manifest_fingerprint": (readiness_permit.support_manifest_fingerprint),
        "model_input_fingerprint": readiness_permit.model_input_fingerprint,
        "input_contract_version": readiness_permit.input_contract_version,
        "support_row_fingerprint": support["support_row_fingerprint"],
        "accepted_taxon_key": _required_text(
            support["accepted_taxon_key"], field="accepted_taxon_key"
        ),
        "scientific_name": _required_text(
            support["scientific_name"], field="scientific_name"
        ),
        "geo_cluster_id": _required_text(
            support["geo_cluster_id"], field="geo_cluster_id"
        ),
        "life_stage": _required_text(support["life_stage"], field="life_stage"),
        "visual_domain": _required_text(
            support["visual_domain"], field="visual_domain"
        ),
        "view": _required_text(support["view"], field="view"),
        "route": _required_text(support["route"], field="route"),
        "source_object_uri": support["source_object_uri"],
        "source_image_sha256": support["image_sha256"],
        "source_object_fingerprint": support["object_fingerprint"],
        "visual_input_id": visual_input.visual_input_id,
        "visual_input_kind": visual_input.visual_input_kind,
        "raw_image_content_hash": visual_input.raw_image_content_hash,
        "image_content_hash": visual_input.image_content_hash,
        "transformation_version": visual_input.transformation_version,
        "transformation_policy_fingerprint": (
            visual_input.transformation_policy_fingerprint
        ),
        "transformation_fingerprint": visual_input.transformation_fingerprint,
        "model_input_schema_version": readiness_permit.model_input_schema_version,
        "model_name": readiness_permit.model_name,
        "model_version": readiness_permit.model_version,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_checkpoint_uri": readiness_permit.checkpoint_uri,
        "model_weights_sha256": model_weights_sha256,
        "model_checkpoint_hash": model_weights_sha256,
        "model_fingerprint": _reference_embedding_model_fingerprint(
            model_input_fingerprint=readiness_permit.model_input_fingerprint,
            embedding_dimension=embedding_dimension,
        ),
        "preprocessing_version": preprocessing_contract.version,
        "preprocessing_fingerprint": preprocessing_contract.fingerprint,
        "open_clip_version": preprocessing_attestation.open_clip_version,
        "open_clip_config_sha256": (preprocessing_attestation.open_clip_config_sha256),
        "preprocessing_attestation_version": (
            preprocessing_attestation.preprocessing_version
        ),
        "preprocessing_config_json": (
            preprocessing_attestation.preprocessing_config_json
        ),
        "preprocessing_attestation_fingerprint": (
            preprocessing_attestation.preprocessing_fingerprint
        ),
        "embedding_dimension": embedding_dimension,
        "embedding": list(vector),
        "embedding_norm": embedding_norm,
        "support_split": _required_text(
            support["support_split"], field="support_split"
        ),
        "embedding_created_at": embedding_created_at,
        "embedding_fingerprint": "",
    }


def _stored_unit_vector(values: Sequence[float]) -> tuple[tuple[float, ...], float]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("reference embedding vector must be non-empty")
    source: list[float] = []
    for raw_value in values:
        if isinstance(raw_value, bool):
            raise ValueError("reference embedding vector must contain finite values")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "reference embedding vector must contain finite values"
            ) from exc
        if not isfinite(value):
            raise ValueError("reference embedding vector must contain finite values")
        source.append(value)
    if not source:
        raise ValueError("reference embedding vector must be non-empty")
    try:
        stored = tuple(float(value) for value in array("f", source))
    except OverflowError as exc:
        raise ValueError(
            "reference embedding vector must contain finite Float32 values"
        ) from exc
    if any(not isfinite(value) for value in stored):
        raise ValueError(
            "reference embedding vector must contain finite Float32 values"
        )
    norm = sqrt(sum(value * value for value in stored))
    if not isfinite(norm) or norm <= 0:
        raise ValueError("reference embedding vector must have non-zero norm")
    if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
        raise ValueError("reference embedding vector must be unit-normalized")
    return stored, norm


def _reference_embedding_model_fingerprint(
    *,
    model_input_fingerprint: str,
    embedding_dimension: int,
) -> str:
    return _json_fingerprint(
        {
            "schema_version": REFERENCE_EMBEDDING_MODEL_FINGERPRINT_SCHEMA_VERSION,
            "model_input_fingerprint": _sha256(
                model_input_fingerprint,
                field="model_input_fingerprint",
            ),
            "embedding_dimension": _positive_dimension(embedding_dimension),
            "embedding_dtype": REFERENCE_EMBEDDING_DTYPE,
            "normalization_policy": REFERENCE_EMBEDDING_NORMALIZATION_POLICY,
        }
    )


def _embedding_row_fingerprint_preimage(row: Mapping[str, object]) -> bytes:
    vector = tuple(float(value) for value in row["embedding"])
    _, preprocessing_config = _canonical_preprocessing_config(
        json.loads(
            _required_text(
                row["preprocessing_config_json"],
                field="preprocessing_config_json",
            )
        )
    )
    semantic = {
        "schema_version": row["schema_version"],
        "registry_version": row["registry_version"],
        "reference_bank_version": row["reference_bank_version"],
        "reference_media_id": row["reference_media_id"],
        "reference_observation_id": row["reference_observation_id"],
        "source_snapshot_version": row["source_snapshot_version"],
        "duplicate_group_id": row["duplicate_group_id"],
        "reference_bank_fingerprint": row["reference_bank_fingerprint"],
        "support_manifest_fingerprint": row["support_manifest_fingerprint"],
        "model_input_fingerprint": row["model_input_fingerprint"],
        "input_contract_version": row["input_contract_version"],
        "support_row_fingerprint": row["support_row_fingerprint"],
        "accepted_taxon_key": row["accepted_taxon_key"],
        "scientific_name": row["scientific_name"],
        "geo_cluster_id": row["geo_cluster_id"],
        "life_stage": row["life_stage"],
        "visual_domain": row["visual_domain"],
        "view": row["view"],
        "route": row["route"],
        "source_image_sha256": row["source_image_sha256"],
        "visual_input_id": row["visual_input_id"],
        "visual_input_kind": row["visual_input_kind"],
        "raw_image_content_hash": row["raw_image_content_hash"],
        "image_content_hash": row["image_content_hash"],
        "transformation_version": row["transformation_version"],
        "transformation_policy_fingerprint": row["transformation_policy_fingerprint"],
        "transformation_fingerprint": row["transformation_fingerprint"],
        "model_input_schema_version": row["model_input_schema_version"],
        "model_name": row["model_name"],
        "model_version": row["model_version"],
        "model_id": row["model_id"],
        "model_revision": row["model_revision"],
        "model_weights_sha256": row["model_weights_sha256"],
        "model_checkpoint_hash": row["model_checkpoint_hash"],
        "model_fingerprint": row["model_fingerprint"],
        "preprocessing_version": row["preprocessing_version"],
        "preprocessing_fingerprint": row["preprocessing_fingerprint"],
        "open_clip_version": row["open_clip_version"],
        "open_clip_config_sha256": row["open_clip_config_sha256"],
        "preprocessing_attestation_version": row["preprocessing_attestation_version"],
        "preprocessing_config": preprocessing_config,
        "preprocessing_attestation_fingerprint": row[
            "preprocessing_attestation_fingerprint"
        ],
        "embedding_dimension": int(row["embedding_dimension"]),
        "support_split": row["support_split"],
    }
    encoded = canonical_semantic_bytes(semantic)
    preimage = bytearray()
    preimage.extend(len(encoded).to_bytes(8, "big"))
    preimage.extend(encoded)
    preimage.extend(struct.pack("<d", float(row["embedding_norm"])))
    for value in vector:
        preimage.extend(struct.pack("<f", value))
    return bytes(preimage)


def _embedding_row_fingerprint(row: Mapping[str, object]) -> str:
    return (
        "sha256:" + hashlib.sha256(_embedding_row_fingerprint_preimage(row)).hexdigest()
    )


def _publication_report(
    frame: pl.DataFrame,
    *,
    artifact_uri: str,
    artifact_byte_count: int,
    artifact_sha256: str,
    run_id: str,
    git_sha: str,
    pid: int,
    started_at: datetime,
    ended_at: datetime,
    command: str = "bioclip.publish_reference_embeddings",
    worker_id: str | None = None,
) -> dict[str, object]:
    dimension = int(frame["embedding_dimension"][0])
    report: dict[str, object] = {
        "schema_version": REFERENCE_EMBEDDINGS_REPORT_SCHEMA_VERSION,
        "command": _required_text(command, field="command"),
        "run_id": run_id,
        "pid": _positive_integer(pid, field="pid"),
        "git_sha": _provenance_git_sha(git_sha),
        "status": "completed",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": (ended_at - started_at).total_seconds(),
        "row_count": frame.height,
        "input_row_count": frame.height,
        "output_row_count": frame.height,
        "embedding_dimension": dimension,
        "model_id": str(frame["model_id"][0]),
        "model_revision": str(frame["model_revision"][0]),
        "model_checkpoint_uri": str(frame["model_checkpoint_uri"][0]),
        "model_input_fingerprint": str(frame["model_input_fingerprint"][0]),
        "input_contract_version": str(frame["input_contract_version"][0]),
        "model_weights_sha256": str(frame["model_weights_sha256"][0]),
        "readiness_sha256": str(frame["readiness_sha256"][0]),
        "reference_bank_fingerprint": str(frame["reference_bank_fingerprint"][0]),
        "support_manifest_fingerprint": str(frame["support_manifest_fingerprint"][0]),
        "preprocessing_version": str(frame["preprocessing_version"][0]),
        "preprocessing_fingerprint": str(frame["preprocessing_fingerprint"][0]),
        "open_clip_version": str(frame["open_clip_version"][0]),
        "open_clip_config_sha256": str(frame["open_clip_config_sha256"][0]),
        "preprocessing_attestation_version": str(
            frame["preprocessing_attestation_version"][0]
        ),
        "preprocessing_attestation_fingerprint": str(
            frame["preprocessing_attestation_fingerprint"][0]
        ),
        "support_split_counts": _value_counts(frame, "support_split"),
        "visual_input_kind_counts": _value_counts(frame, "visual_input_kind"),
        "artifact_fingerprint": reference_embeddings_artifact_fingerprint(frame),
        "retry_count": None,
        "error_count": 0,
        "gpu_memory_bytes": "not_instrumented",
        "peak_rss_bytes": "not_instrumented",
        "artifact": {
            "uri": artifact_uri,
            "byte_count": artifact_byte_count,
            "sha256": artifact_sha256,
        },
    }
    if worker_id is not None:
        report["worker_id"] = _required_text(worker_id, field="worker_id")
    return report


def _publication_report_semantic_fingerprint(
    report: Mapping[str, object],
) -> str:
    return _json_fingerprint(
        {
            "schema_version": report["schema_version"],
            "status": report["status"],
            "row_count": report["row_count"],
            "input_row_count": report["input_row_count"],
            "output_row_count": report["output_row_count"],
            "embedding_dimension": report["embedding_dimension"],
            "support_split_counts": report["support_split_counts"],
            "visual_input_kind_counts": report["visual_input_kind_counts"],
            "artifact_fingerprint": report["artifact_fingerprint"],
        }
    )


def _publication_summary_semantic_fingerprint(
    report: Mapping[str, object],
) -> str:
    return _json_fingerprint(
        {
            "schema_version": REFERENCE_EMBEDDINGS_SUMMARY_SCHEMA_VERSION,
            "report_semantic_fingerprint": (
                _publication_report_semantic_fingerprint(report)
            ),
        }
    )


def _publication_markdown(report: Mapping[str, object]) -> str:
    artifact = report["artifact"]
    assert isinstance(artifact, Mapping)
    return "\n".join(
        (
            "# Reference embedding build",
            "",
            f"- Status: {report['status']}",
            f"- Run ID: {report['run_id']}",
            f"- Rows: {report['row_count']}",
            f"- Embedding dimension: {report['embedding_dimension']}",
            f"- Model: {report['model_id']} @ {report['model_revision']}",
            f"- Artifact: {artifact['uri']}",
            f"- Artifact bytes: {artifact['byte_count']}",
            f"- Artifact SHA-256: {artifact['sha256']}",
            "",
        )
    )


def _value_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in frame[column]).items()))


def _require_single_value(
    frame: pl.DataFrame,
    column: str,
    expected: str | None,
) -> str:
    values = {str(value or "") for value in frame[column].to_list()}
    if len(values) != 1 or not next(iter(values), ""):
        raise ValueError(f"reference embeddings have mixed or blank {column}")
    value = next(iter(values))
    if expected is not None and value != expected:
        raise ValueError(f"reference embeddings {column} mismatch")
    return value


def _required_scorer_text(
    scorer: ReferenceImageEmbeddingScorer,
    field: str,
) -> str:
    return _required_text(getattr(scorer, field, None), field=field)


def _required_text(value: object, *, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field} must be non-empty")
    return result


def _sha256(value: object, *, field: str) -> str:
    result = str(value or "")
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{field} must be a lowercase sha256 fingerprint")
    return result


def _validate_absolute_uri(value: object, *, field: str) -> str:
    uri = _required_text(value, field=field)
    parsed = urlsplit(uri)
    if not parsed.scheme or (not parsed.netloc and not parsed.path.startswith("/")):
        raise ValueError(f"{field} must be an absolute URI")
    return uri


def _json_fingerprint(value: object) -> str:
    return canonical_semantic_fingerprint(value)


def _write_json_atomically(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    value,
                    sort_keys=True,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _positive_dimension(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("embedding dimension must be a positive integer")
    if value <= 0:
        raise ValueError("embedding dimension must be a positive integer")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _provenance_git_sha(value: object) -> str:
    result = str(value or current_git_sha() or "not_instrumented").strip()
    if result == "not_instrumented" or _GIT_SHA_PATTERN.fullmatch(result):
        return result
    raise ValueError("git_sha must be a Git object ID or not_instrumented")


def _utc_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _log_event(event: str, **fields: object) -> None:
    _LOGGER.info(
        "%s",
        json.dumps(
            {"event": event, **fields},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
    )


__all__ = [
    "REFERENCE_EMBEDDINGS_CHECKPOINT_PARTS_DIR",
    "REFERENCE_EMBEDDINGS_CHECKPOINT_LOCK_SUFFIX",
    "REFERENCE_EMBEDDINGS_CHECKPOINT_SCHEMA_VERSION",
    "REFERENCE_EMBEDDINGS_CHECKPOINT_STATE_FILE",
    "REFERENCE_EMBEDDINGS_FILE",
    "REFERENCE_EMBEDDINGS_MANIFEST_FILE",
    "REFERENCE_EMBEDDINGS_MANIFEST_SCHEMA_VERSION",
    "REFERENCE_EMBEDDINGS_REPORT_FILE",
    "REFERENCE_EMBEDDINGS_REPORT_SCHEMA_VERSION",
    "REFERENCE_EMBEDDINGS_SCHEMA_VERSION",
    "REFERENCE_EMBEDDINGS_SUMMARY_FILE",
    "ReferenceImageEmbeddingScorer",
    "ReferenceEmbeddingCheckpointBusyError",
    "ReferenceVisualInput",
    "build_reference_embeddings",
    "decoded_image_file_content_hash",
    "load_reference_embeddings",
    "publish_reference_embeddings",
    "publish_reference_embeddings_to_cloud",
    "reference_embeddings_artifact_fingerprint",
    "reference_embeddings_schema",
    "validate_reference_embeddings",
    "write_reference_embeddings",
]
