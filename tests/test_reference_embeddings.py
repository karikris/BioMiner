from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
from math import isclose
from pathlib import Path
import struct
from threading import Event
from urllib.parse import unquote, urlsplit

from PIL import Image
import polars as pl
from polars.testing import assert_frame_equal
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import biominer.bioclip.reference_embeddings as reference_embeddings_module
from biominer.bioclip.reference_embeddings import (
    REFERENCE_EMBEDDINGS_CHECKPOINT_PARTS_DIR,
    REFERENCE_EMBEDDINGS_CHECKPOINT_STATE_FILE,
    REFERENCE_EMBEDDINGS_FILE,
    REFERENCE_EMBEDDINGS_MANIFEST_FILE,
    REFERENCE_EMBEDDINGS_REPORT_FILE,
    REFERENCE_EMBEDDINGS_SUMMARY_FILE,
    ReferenceEmbeddingCheckpointBusyError,
    ReferenceVisualInput,
    build_reference_embeddings as _build_reference_embeddings,
    decoded_image_file_content_hash,
    load_reference_embeddings,
    publish_reference_embeddings,
    publish_reference_embeddings_to_cloud,
    reference_embeddings_artifact_fingerprint,
    reference_embeddings_schema,
    validate_reference_embeddings,
    write_reference_embeddings,
)
from biominer.bioclip.object_runner import EphemeralCropBioClipScorer
import biominer.references.readiness as readiness_module
from biominer.references.readiness import (
    REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION,
    ReferenceBankReadinessPermit,
    ReferenceModelInputIdentity,
    make_reference_split_assignment_fingerprint,
    reference_support_manifest_fingerprint,
    reference_support_manifest_schema,
)
from biominer.reports.flickr_fetch import current_git_sha
from biominer.detection.detector_base import DecodedImage
from biominer.vision.full_frame_attention import (
    AttentionRegion,
    FOCUSED_FULL_FRAME_KIND,
    FULL_FRAME_VISUAL_INPUT_VERSION,
    FullFrameAttentionVariant,
    RAW_FULL_IMAGE_KIND,
    TargetPreprocessingContract,
    generate_full_frame_attention_variants,
    raw_full_frame_visual_input,
)
from biominer.storage.local import LocalStorageBackend
from biominer.storage.uri import join_uri
from biominer.workstore.sqlite import SQLiteWorkStore


NOW = datetime(2026, 7, 14, 7, 0, tzinfo=UTC)
REVISION = "191d741545e4c741cdef4b22c6eb69c945c1e592"
WEIGHTS_SHA256 = "sha256:" + "f" * 64
OPEN_CLIP_CONFIG_SHA256 = "sha256:" + "e" * 64
PREPROCESSING_ATTESTATION_VERSION = "openclip-preprocessing-attestation-v2"
PREPROCESSING_CONFIG = {
    "fill_color": 0,
    "interpolation": "bicubic",
    "mean": [0.48145466, 0.4578275, 0.40821073],
    "mode": "RGB",
    "resize_mode": "longest",
    "size": [224, 224],
    "std": [0.26862954, 0.26130258, 0.27577711],
}


class FakeScorer:
    model_id = "imageomics/bioclip-2.5-vith14"
    model_revision = REVISION
    model_weights_sha256: str | None = WEIGHTS_SHA256
    image_resize_mode = "longest"
    effective_image_resize_mode = "longest"
    open_clip_version = "3.3.0"
    open_clip_config_sha256 = OPEN_CLIP_CONFIG_SHA256
    preprocessing_version = PREPROCESSING_ATTESTATION_VERSION
    preprocessing_config = PREPROCESSING_CONFIG
    preprocessing_fingerprint = ""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[Path, ...]] = []
        self.close_calls = 0
        self.attestation_calls = 0
        self.last_image_content_hashes: tuple[str, ...] = ()
        self.preprocessing_config = dict(PREPROCESSING_CONFIG)
        self.preprocessing_fingerprint = _preprocessing_attestation_fingerprint(
            self.preprocessing_config
        )

    def embed_image_paths(self, image_paths: list[Path]) -> list[list[float]]:
        paths = tuple(image_paths)
        self.calls.append(paths)
        self.last_image_content_hashes = tuple(
            decoded_image_file_content_hash(path) for path in paths
        )
        return [self.vectors[path.name] for path in paths]

    def close(self) -> None:
        self.close_calls += 1

    def ensure_model_attestation(self) -> None:
        self.attestation_calls += 1


def build_reference_embeddings(
    support_manifest: pl.DataFrame,
    visual_inputs: list[ReferenceVisualInput],
    *,
    scorer: FakeScorer,
    **kwargs: object,
) -> pl.DataFrame:
    return _build_reference_embeddings(
        support_manifest,
        visual_inputs,
        readiness_permit=_permit(support_manifest),
        scorer=scorer,
        **kwargs,
    )


def test_build_persists_all_eligible_splits_and_reuses_one_scorer(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(
        tmp_path,
        (
            ("media-train", "support_train"),
            ("media-select", "model_selection"),
            ("media-calibrate", "calibration"),
            ("media-test", "final_test"),
        ),
    )
    inputs = _visual_inputs(tmp_path, manifest)
    train_row = manifest.filter(pl.col("reference_media_id") == "media-train").row(
        0,
        named=True,
    )
    source_path = Path(unquote(urlsplit(str(train_row["source_object_uri"])).path))
    source_image = _decoded_image(source_path)
    attention = generate_full_frame_attention_variants(
        source_image,
        [
            AttentionRegion(
                source_detection_id="detection:fixture",
                route="adult_field",
                bbox_xyxyn=(0.1, 0.1, 0.8, 0.9),
            )
        ],
        source_type="reference_fixture",
    )
    focused = next(
        variant
        for variant in attention.variants
        if variant.visual_input_kind == FOCUSED_FULL_FRAME_KIND
    )
    focused_path = tmp_path / "media-train-focused.png"
    Image.frombytes(
        focused.image.mode,
        (focused.image.width, focused.image.height),
        focused.image.data,
    ).save(focused_path, format="PNG")
    inputs.append(
        ReferenceVisualInput.from_variant(
            reference_media_id="media-train",
            source_image_path=source_path,
            image_path=focused_path,
            variant=focused,
        )
    )
    vectors = {
        item.image_path.name: _vector(index)
        for index, item in enumerate(
            sorted(inputs, key=lambda item: item.image_path.name)
        )
    }
    scorer = FakeScorer(vectors)

    frame = build_reference_embeddings(
        manifest.reverse(),
        list(reversed(inputs)),
        scorer=scorer,
        batch_size=2,
        embedding_created_at=NOW,
    )

    assert frame.schema == reference_embeddings_schema(3)
    assert frame.height == 5
    assert set(frame["support_split"]) == {
        "support_train",
        "model_selection",
        "calibration",
        "final_test",
    }
    assert set(frame["visual_input_kind"]) == {
        RAW_FULL_IMAGE_KIND,
        FOCUSED_FULL_FRAME_KIND,
    }
    assert frame["model_id"].unique().to_list() == [scorer.model_id]
    assert frame["model_revision"].unique().to_list() == [REVISION]
    assert frame["model_weights_sha256"].unique().to_list() == [WEIGHTS_SHA256]
    assert frame["preprocessing_version"].unique().to_list() == [
        TargetPreprocessingContract().version
    ]
    assert frame["preprocessing_fingerprint"].unique().to_list() == [
        TargetPreprocessingContract().fingerprint
    ]
    assert frame["open_clip_version"].unique().to_list() == ["3.3.0"]
    assert frame["open_clip_config_sha256"].unique().to_list() == [
        OPEN_CLIP_CONFIG_SHA256
    ]
    assert frame["preprocessing_attestation_fingerprint"].unique().to_list() == [
        scorer.preprocessing_fingerprint
    ]
    assert frame["embedding_dimension"].unique().to_list() == [3]
    assert frame["reference_admission_mode"].unique().to_list() == [
        "human_verified_strict"
    ]
    assert frame["admission_policy_fingerprint"].unique().to_list() == [
        _permit(manifest).admission_policy_fingerprint
    ]
    assert frame["identity_evidence_basis"].unique().to_list() == [
        "human_verified"
    ]
    assert frame["provisional_support"].to_list() == [False] * frame.height
    assert frame["human_review_status"].unique().to_list() == ["completed"]
    assert frame["reference_quality_flags"].to_list() == [[]] * frame.height
    assert all(isclose(value, 1.0, abs_tol=1e-6) for value in frame["embedding_norm"])
    assert frame["embedding_fingerprint"].n_unique() == frame.height
    assert [len(call) for call in scorer.calls] == [2, 2]
    path_by_visual_input = {
        item.visual_input_id: item.image_path.name for item in inputs
    }
    seen_content_hashes: set[str] = set()
    expected_paths: list[str] = []
    for row in frame.select(
        "visual_input_id",
        "image_content_hash",
    ).iter_rows(named=True):
        if row["image_content_hash"] in seen_content_hashes:
            continue
        seen_content_hashes.add(row["image_content_hash"])
        expected_paths.append(path_by_visual_input[row["visual_input_id"]])
    assert [path.name for call in scorer.calls for path in call] == expected_paths
    assert scorer.close_calls == 0


def test_crop_runtime_binds_loaded_worker_attestation_to_readiness(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    permit = _permit(manifest)
    persistent = FakeScorer({})
    scorer = EphemeralCropBioClipScorer(
        scorer=persistent,
        image_loader=lambda _record: None,
        temp_dir=tmp_path / "crops",
        model_id=persistent.model_id,
        model_version="unbound",
        model_checkpoint=persistent.model_revision,
    )

    scorer.bind_reference_readiness(permit)

    assert persistent.attestation_calls == 1
    assert scorer.checkpoint_sha256 == permit.checkpoint_sha256
    assert scorer.preprocessing_version == permit.preprocessing_version
    assert scorer.input_contract_version == permit.input_contract_version
    assert scorer.model_input_fingerprint == permit.model_input_fingerprint


def test_build_is_semantically_deterministic_across_order_batching_and_time(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(
        tmp_path, (("media-a", "support_train"), ("media-b", "model_selection"))
    )
    inputs = _visual_inputs(tmp_path, manifest)
    vectors = {
        inputs[0].image_path.name: [1.0, 0.0, 0.0],
        inputs[1].image_path.name: [0.0, 1.0, 0.0],
    }

    first = build_reference_embeddings(
        manifest,
        inputs,
        scorer=FakeScorer(vectors),
        batch_size=1,
        embedding_created_at=NOW,
    )
    second = build_reference_embeddings(
        manifest.reverse(),
        list(reversed(inputs)),
        scorer=FakeScorer(vectors),
        batch_size=20,
        embedding_created_at=NOW + timedelta(days=1),
    )

    assert_frame_equal(
        first.drop("embedding_created_at"),
        second.drop("embedding_created_at"),
        check_exact=True,
    )
    assert first["embedding_created_at"].to_list() == [NOW, NOW]
    assert second["embedding_created_at"].to_list() == [
        NOW + timedelta(days=1),
        NOW + timedelta(days=1),
    ]
    assert reference_embeddings_artifact_fingerprint(first) == (
        reference_embeddings_artifact_fingerprint(second)
    )


def test_build_reuses_durable_content_cache_across_support_manifest_versions(
    tmp_path: Path,
) -> None:
    first_manifest = _support_manifest(
        tmp_path,
        (("media-a", "support_train"),),
    )
    first = build_reference_embeddings(
        first_manifest,
        _visual_inputs(tmp_path, first_manifest),
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )
    cache_path = write_reference_embeddings(first, tmp_path / "cache.parquet")
    second_manifest = _support_manifest(
        tmp_path,
        (
            ("media-a", "support_train"),
            ("media-b", "model_selection"),
        ),
    )
    second_inputs = _visual_inputs(tmp_path, second_manifest)
    scorer = FakeScorer({"media-b.png": [0.0, 1.0, 0.0]})

    second = build_reference_embeddings(
        second_manifest,
        second_inputs,
        scorer=scorer,
        embedding_cache=cache_path,
        embedding_created_at=NOW + timedelta(days=1),
    )

    assert second.height == 2
    assert [path.name for call in scorer.calls for path in call] == ["media-b.png"]
    assert second["support_manifest_fingerprint"].unique().to_list() == [
        _permit(second_manifest).support_manifest_fingerprint
    ]
    media_a = second.filter(pl.col("reference_media_id") == "media-a").row(
        0,
        named=True,
    )
    assert media_a["embedding"] == first.row(0, named=True)["embedding"]
    assert media_a["embedding_created_at"] == NOW + timedelta(days=1)


def test_review_provenance_does_not_change_vector_cache_identity(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    frame = build_reference_embeddings(
        manifest,
        _visual_inputs(tmp_path, manifest),
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )
    original = frame.row(0, named=True)
    changed = {
        **original,
        "human_review_status": "pending",
        "reference_quality_flags": ["review_status_changed"],
    }

    assert reference_embeddings_module._embedding_cache_key_from_row(  # noqa: SLF001 - verifies the normative cache boundary.
        original
    ) == reference_embeddings_module._embedding_cache_key_from_row(changed)  # noqa: SLF001
    assert reference_embeddings_module._embedding_row_fingerprint(  # noqa: SLF001 - provenance must still alter artifact identity.
        original
    ) != reference_embeddings_module._embedding_row_fingerprint(changed)  # noqa: SLF001


def test_durable_embedding_cache_and_artifact_identity_survive_object_relocation(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    first = build_reference_embeddings(
        manifest,
        _visual_inputs(tmp_path, manifest),
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )
    cache_path = write_reference_embeddings(first, tmp_path / "cache.parquet")
    original_source = Path(
        unquote(urlsplit(str(manifest["source_object_uri"][0])).path)
    )
    relocated_source = tmp_path / "relocated" / "media-a.png"
    relocated_source.parent.mkdir(parents=True)
    relocated_source.write_bytes(original_source.read_bytes())
    relocated_manifest = manifest.with_columns(
        pl.lit(relocated_source.resolve().as_uri()).alias("source_object_uri")
    )
    scorer = FakeScorer({})

    relocated = build_reference_embeddings(
        relocated_manifest,
        _visual_inputs(tmp_path, relocated_manifest),
        scorer=scorer,
        embedding_cache=cache_path,
        embedding_created_at=NOW,
    )

    assert scorer.calls == []
    assert reference_embeddings_artifact_fingerprint(relocated) == (
        reference_embeddings_artifact_fingerprint(first)
    )
    assert relocated["source_object_uri"].to_list() == [
        relocated_source.resolve().as_uri()
    ]


def test_regenerated_support_audit_fields_preserve_cache_and_checkpoint_identity(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)
    checkpoint_dir = tmp_path / "checkpoint"
    original = build_reference_embeddings(
        manifest,
        inputs,
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
        checkpoint_dir=checkpoint_dir,
    )
    cache_path = write_reference_embeddings(original, tmp_path / "cache.parquet")
    checkpoint_state_path = checkpoint_dir / REFERENCE_EMBEDDINGS_CHECKPOINT_STATE_FILE
    original_state = json.loads(checkpoint_state_path.read_text(encoding="utf-8"))

    regenerated_rows: list[dict[str, object]] = []
    for source_row in manifest.iter_rows(named=True):
        row = dict(source_row)
        original_support_fingerprint = row["support_row_fingerprint"]
        row.update(
            {
                "review_request_id": "review:republished",
                "review_decision_ids": ["decision:republished"],
                "reviewer_ids": ["reviewer:republished"],
                "source_record_url": (
                    "https://republished.example.test/records/media-a"
                ),
                "licence_uri": ("https://republished.example.test/licences/cc-by-4.0"),
                "object_fingerprint": _sha("republished-object"),
                "split_assignment_fingerprint": _sha("republished-split-assignment"),
            }
        )
        row["support_row_fingerprint"] = readiness_module._support_row_fingerprint(  # noqa: SLF001 - fixture mirrors the persisted semantic identity.
            row
        )
        assert row["support_row_fingerprint"] == original_support_fingerprint
        regenerated_rows.append(row)
    regenerated_manifest = pl.DataFrame(
        regenerated_rows,
        schema=reference_support_manifest_schema(),
        orient="row",
        strict=True,
    )
    original_permit = _permit(manifest)
    regenerated_permit = replace(
        _permit(regenerated_manifest),
        readiness_sha256=_sha("republished-readiness"),
    )
    assert regenerated_permit.support_manifest_fingerprint == (
        original_permit.support_manifest_fingerprint
    )

    checkpoint_scorer = FakeScorer({})
    resumed = _build_reference_embeddings(
        regenerated_manifest,
        inputs,
        readiness_permit=regenerated_permit,
        scorer=checkpoint_scorer,
        embedding_created_at=NOW + timedelta(days=1),
        checkpoint_dir=checkpoint_dir,
    )
    assert checkpoint_scorer.calls == []
    assert json.loads(checkpoint_state_path.read_text(encoding="utf-8")) == (
        original_state
    )

    cache_scorer = FakeScorer({})
    cached = _build_reference_embeddings(
        regenerated_manifest,
        inputs,
        readiness_permit=regenerated_permit,
        scorer=cache_scorer,
        embedding_created_at=NOW + timedelta(days=2),
        embedding_cache=cache_path,
    )
    assert cache_scorer.calls == []
    assert reference_embeddings_artifact_fingerprint(resumed) == (
        reference_embeddings_artifact_fingerprint(original)
    )
    assert reference_embeddings_artifact_fingerprint(cached) == (
        reference_embeddings_artifact_fingerprint(original)
    )
    assert resumed["review_decision_ids"].to_list() == [["decision:republished"]]
    assert cached["source_object_fingerprint"].to_list() == [_sha("republished-object")]


def test_embedding_fingerprint_pins_little_endian_float32_encoding(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)

    frame = build_reference_embeddings(
        manifest,
        inputs,
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )

    row = frame.row(0, named=True)
    preimage = reference_embeddings_module._embedding_row_fingerprint_preimage(  # noqa: SLF001 - verifies the normative binary fingerprint contract.
        row
    )
    little_endian_values = struct.pack("<d", float(row["embedding_norm"])) + b"".join(
        struct.pack("<f", float(value)) for value in row["embedding"]
    )
    big_endian_values = struct.pack(">d", float(row["embedding_norm"])) + b"".join(
        struct.pack(">f", float(value)) for value in row["embedding"]
    )
    preprocessing_mean = float(PREPROCESSING_CONFIG["mean"][0])

    assert preimage.endswith(little_endian_values)
    assert not preimage.endswith(big_endian_values)
    assert struct.pack("<d", preprocessing_mean) in preimage
    assert str(preprocessing_mean).encode("ascii") not in preimage
    assert (
        row["embedding_fingerprint"] == "sha256:" + hashlib.sha256(preimage).hexdigest()
    )


def test_embedding_row_semantic_fingerprint_excludes_operational_audit_provenance(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    frame = build_reference_embeddings(
        manifest,
        _visual_inputs(tmp_path, manifest),
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )
    row = frame.row(0, named=True)
    relocated = dict(row)
    relocated["source_object_uri"] = "s3://relocated/source/media-a.png"
    relocated["model_checkpoint_uri"] = "s3://relocated/models/model.safetensors"
    relocated["readiness_sha256"] = _sha("relocated-readiness-object")
    relocated["review_decision_ids"] = ["decision:republished"]
    relocated["source_object_fingerprint"] = _sha("republished-object")

    assert reference_embeddings_module._embedding_row_fingerprint(  # noqa: SLF001 - verifies normative semantic exclusions.
        relocated
    ) == reference_embeddings_module._embedding_row_fingerprint(row)  # noqa: SLF001


def test_build_order_is_stable_when_distinct_transforms_have_identical_pixels(
    tmp_path: Path,
) -> None:
    source_path = _image(tmp_path / "source" / "media-a.png", (0, 0, 0))
    row = _support_row(
        "media-a",
        source_path=source_path,
        support_split="support_train",
        eligible=True,
    )
    manifest = pl.DataFrame(
        [row],
        schema=reference_support_manifest_schema(),
        orient="row",
    )
    source_image = _decoded_image(source_path)
    generated = generate_full_frame_attention_variants(
        source_image,
        (
            AttentionRegion(
                source_detection_id="detection:a",
                route="adult_field",
                bbox_xyxyn=(0.0, 0.0, 0.5, 0.5),
            ),
            AttentionRegion(
                source_detection_id="detection:b",
                route="adult_field",
                bbox_xyxyn=(0.5, 0.5, 1.0, 1.0),
            ),
        ),
        source_type="reference_fixture",
    )
    focused = [
        variant
        for variant in generated.variants
        if variant.visual_input_kind == FOCUSED_FULL_FRAME_KIND
    ]
    assert len(focused) == 2
    assert len({variant.visual_content_hash for variant in focused}) == 1
    raw_variant = raw_full_frame_visual_input(source_image)
    inputs: list[ReferenceVisualInput] = [
        ReferenceVisualInput.from_variant(
            reference_media_id="media-a",
            source_image_path=source_path,
            image_path=source_path,
            variant=raw_variant,
        )
    ]
    for index, variant in enumerate(focused):
        path = tmp_path / f"focused-{index}.png"
        Image.frombytes(
            variant.image.mode,
            (variant.image.width, variant.image.height),
            variant.image.data,
        ).save(path, format="PNG")
        inputs.append(
            ReferenceVisualInput.from_variant(
                reference_media_id="media-a",
                source_image_path=source_path,
                image_path=path,
                variant=variant,
            )
        )
    vectors = {item.image_path.name: [1.0, 0.0, 0.0] for item in inputs}
    assert len({item.image_content_hash for item in inputs}) == 1
    first_scorer = FakeScorer(vectors)
    second_scorer = FakeScorer(vectors)

    first = build_reference_embeddings(
        manifest,
        inputs,
        scorer=first_scorer,
        embedding_created_at=NOW,
    )
    second = build_reference_embeddings(
        manifest,
        list(reversed(inputs)),
        scorer=second_scorer,
        embedding_created_at=NOW,
    )

    assert_frame_equal(first, second, check_exact=True)
    assert reference_embeddings_artifact_fingerprint(first) == (
        reference_embeddings_artifact_fingerprint(second)
    )
    assert sum(len(call) for call in first_scorer.calls) == 1
    assert sum(len(call) for call in second_scorer.calls) == 1


def test_build_resumes_durable_batches_without_reembedding_completed_inputs(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(
        tmp_path,
        (
            ("media-a", "support_train"),
            ("media-b", "support_train"),
            ("media-c", "support_train"),
        ),
    )
    inputs = _visual_inputs(tmp_path, manifest)
    vectors = {
        "media-a.png": [1.0, 0.0, 0.0],
        "media-b.png": [0.0, 1.0, 0.0],
        "media-c.png": [0.0, 0.0, 1.0],
    }

    class FailSecondBatch(FakeScorer):
        def embed_image_paths(self, image_paths: list[Path]) -> list[list[float]]:
            if self.calls:
                raise RuntimeError("fixture interruption")
            return super().embed_image_paths(image_paths)

    fresh = build_reference_embeddings(
        manifest,
        inputs,
        scorer=FakeScorer(vectors),
        batch_size=3,
        embedding_created_at=NOW,
    )
    checkpoint_dir = tmp_path / "checkpoint"
    with pytest.raises(RuntimeError, match="fixture interruption"):
        build_reference_embeddings(
            manifest,
            inputs,
            scorer=FailSecondBatch(vectors),
            batch_size=1,
            embedding_created_at=NOW,
            checkpoint_dir=checkpoint_dir,
        )
    inputs[0].image_path.unlink()
    inputs[0].source_image_path.unlink()

    resumed_scorer = FakeScorer(vectors)
    resumed = build_reference_embeddings(
        manifest,
        inputs,
        scorer=resumed_scorer,
        batch_size=1,
        embedding_created_at=NOW,
        checkpoint_dir=checkpoint_dir,
    )

    assert_frame_equal(resumed, fresh, check_exact=True)
    assert [path.name for call in resumed_scorer.calls for path in call] == [
        "media-b.png",
        "media-c.png",
    ]
    state = json.loads(
        (checkpoint_dir / REFERENCE_EMBEDDINGS_CHECKPOINT_STATE_FILE).read_text(
            encoding="utf-8"
        )
    )
    assert state["row_count"] == 3
    assert len(state["parts"]) == 3
    assert (
        len(
            list(
                (checkpoint_dir / REFERENCE_EMBEDDINGS_CHECKPOINT_PARTS_DIR).glob(
                    "*.parquet"
                )
            )
        )
        == 3
    )


def test_partial_resume_survives_support_relocation_and_readiness_republication(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(
        tmp_path,
        (
            ("media-a", "support_train"),
            ("media-b", "support_train"),
            ("media-c", "support_train"),
        ),
    )
    inputs = _visual_inputs(tmp_path, manifest)
    vectors = {
        "media-a.png": [1.0, 0.0, 0.0],
        "media-b.png": [0.0, 1.0, 0.0],
        "media-c.png": [0.0, 0.0, 1.0],
    }

    class FailSecondBatch(FakeScorer):
        def embed_image_paths(self, image_paths: list[Path]) -> list[list[float]]:
            if self.calls:
                raise RuntimeError("fixture interruption")
            return super().embed_image_paths(image_paths)

    checkpoint_dir = tmp_path / "relocatable-checkpoint"
    original_permit = _permit(manifest)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        _build_reference_embeddings(
            manifest,
            inputs,
            readiness_permit=original_permit,
            scorer=FailSecondBatch(vectors),
            batch_size=1,
            embedding_created_at=NOW,
            checkpoint_dir=checkpoint_dir,
        )

    relocated_rows: list[dict[str, object]] = []
    relocated_sources: dict[str, Path] = {}
    original_fingerprints = {
        str(row["reference_media_id"]): str(row["support_row_fingerprint"])
        for row in manifest.iter_rows(named=True)
    }
    for row in manifest.iter_rows(named=True):
        media_id = str(row["reference_media_id"])
        original_source = Path(unquote(urlsplit(str(row["source_object_uri"])).path))
        relocated_source = tmp_path / "relocated" / f"{media_id}.png"
        relocated_source.parent.mkdir(parents=True, exist_ok=True)
        relocated_source.write_bytes(original_source.read_bytes())
        relocated_sources[media_id] = relocated_source
        row["source_object_uri"] = relocated_source.resolve().as_uri()
        row["support_row_fingerprint"] = readiness_module._support_row_fingerprint(row)  # noqa: SLF001 - fixture mirrors the persisted semantic identity.
        assert row["support_row_fingerprint"] == original_fingerprints[media_id]
        relocated_rows.append(row)
    relocated_manifest = pl.DataFrame(
        relocated_rows,
        schema=reference_support_manifest_schema(),
        orient="row",
        strict=True,
    )
    relocated_inputs = [
        replace(
            item,
            source_image_path=relocated_sources[item.reference_media_id],
        )
        for item in inputs
    ]
    completed_input = next(
        item for item in relocated_inputs if item.reference_media_id == "media-a"
    )
    completed_input.image_path.unlink()
    republished_permit = replace(
        _permit(relocated_manifest),
        readiness_sha256=_sha("republished-readiness-object"),
    )
    assert republished_permit.readiness_sha256 != original_permit.readiness_sha256
    assert (
        republished_permit.support_manifest_fingerprint
        == original_permit.support_manifest_fingerprint
    )

    resumed_scorer = FakeScorer(
        {"media-b.png": vectors["media-b.png"], "media-c.png": vectors["media-c.png"]}
    )
    resumed = _build_reference_embeddings(
        relocated_manifest,
        relocated_inputs,
        readiness_permit=republished_permit,
        scorer=resumed_scorer,
        batch_size=1,
        embedding_created_at=NOW + timedelta(days=1),
        checkpoint_dir=checkpoint_dir,
    )

    assert [path.name for call in resumed_scorer.calls for path in call] == [
        "media-b.png",
        "media-c.png",
    ]
    assert resumed["readiness_sha256"].unique().to_list() == [
        republished_permit.readiness_sha256
    ]
    assert resumed["embedding_created_at"].unique().to_list() == [NOW]
    assert set(resumed["source_object_uri"].to_list()) == {
        path.resolve().as_uri() for path in relocated_sources.values()
    }

    fully_resumed_scorer = FakeScorer({})
    fully_resumed = _build_reference_embeddings(
        relocated_manifest,
        relocated_inputs,
        readiness_permit=republished_permit,
        scorer=fully_resumed_scorer,
        batch_size=1,
        embedding_created_at=NOW + timedelta(days=2),
        checkpoint_dir=checkpoint_dir,
    )
    assert_frame_equal(fully_resumed, resumed, check_exact=True)
    assert fully_resumed_scorer.calls == []


def test_resume_accepts_partial_checkpoint_with_transformed_only_media(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(
        tmp_path,
        (
            ("media-a", "support_train"),
            ("media-b", "support_train"),
        ),
    )
    inputs = _visual_inputs(tmp_path, manifest)
    media_b = manifest.filter(pl.col("reference_media_id") == "media-b").row(
        0,
        named=True,
    )
    media_b_source = Path(unquote(urlsplit(str(media_b["source_object_uri"])).path))
    raw_a = next(item for item in inputs if item.reference_media_id == "media-a")
    raw_b = next(item for item in inputs if item.reference_media_id == "media-b")
    focused_image = _decoded_image(raw_a.image_path)
    focused_fingerprint = _sha("focused-transform")
    focused_variant = FullFrameAttentionVariant(
        visual_input_id=reference_embeddings_module._expected_visual_input_id_from_values(  # noqa: SLF001 - constructs the adversarial shared-content fixture.
            visual_input_kind=FOCUSED_FULL_FRAME_KIND,
            raw_image_content_hash=raw_b.raw_image_content_hash,
            image_content_hash=raw_a.image_content_hash,
            transformation_fingerprint=focused_fingerprint,
        ),
        visual_input_kind=FOCUSED_FULL_FRAME_KIND,
        visual_input_version=FULL_FRAME_VISUAL_INPUT_VERSION,
        raw_image_content_hash=raw_b.raw_image_content_hash,
        visual_content_hash=raw_a.image_content_hash,
        transformation_applied=True,
        transformation_version="focused-test-v1",
        transformation_policy_fingerprint=_sha("focused-policy"),
        transformation_fingerprint=focused_fingerprint,
        width=focused_image.width,
        height=focused_image.height,
        mode=focused_image.mode,
        image=focused_image,
    )
    focused_path = tmp_path / "media-b-focused.png"
    focused_path.write_bytes(raw_a.image_path.read_bytes())
    inputs.append(
        ReferenceVisualInput.from_variant(
            reference_media_id="media-b",
            source_image_path=media_b_source,
            image_path=focused_path,
            variant=focused_variant,
        )
    )

    class FailSecondBatch(FakeScorer):
        def embed_image_paths(self, image_paths: list[Path]) -> list[list[float]]:
            if self.calls:
                raise RuntimeError("fixture interruption")
            return super().embed_image_paths(image_paths)

    checkpoint_dir = tmp_path / "partial-checkpoint"
    with pytest.raises(RuntimeError, match="fixture interruption"):
        build_reference_embeddings(
            manifest,
            inputs,
            scorer=FailSecondBatch(
                {
                    "media-a.png": [1.0, 0.0, 0.0],
                    "media-b.png": [0.0, 1.0, 0.0],
                    "media-b-focused.png": [1.0, 0.0, 0.0],
                }
            ),
            batch_size=1,
            checkpoint_dir=checkpoint_dir,
            embedding_created_at=NOW,
        )

    resumed_scorer = FakeScorer(
        {
            "media-b.png": [0.0, 1.0, 0.0],
            "media-b-focused.png": [1.0, 0.0, 0.0],
        }
    )
    resumed = build_reference_embeddings(
        manifest,
        inputs,
        scorer=resumed_scorer,
        batch_size=1,
        checkpoint_dir=checkpoint_dir,
        embedding_created_at=NOW,
    )

    assert resumed.height == 3
    assert [path.name for call in resumed_scorer.calls for path in call] == [
        "media-b.png"
    ]
    validate_reference_embeddings(resumed)


def test_checkpoint_rejects_concurrent_writer_before_model_work(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)
    checkpoint_dir = tmp_path / "checkpoint"
    entered = Event()
    release = Event()

    class BlockingScorer(FakeScorer):
        def embed_image_paths(self, image_paths: list[Path]) -> list[list[float]]:
            entered.set()
            if not release.wait(timeout=10):
                raise RuntimeError("fixture timed out waiting for release")
            return super().embed_image_paths(image_paths)

    first_scorer = BlockingScorer({"media-a.png": [1.0, 0.0, 0.0]})
    second_scorer = FakeScorer({"media-a.png": [1.0, 0.0, 0.0]})
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            build_reference_embeddings,
            manifest,
            inputs,
            scorer=first_scorer,
            checkpoint_dir=checkpoint_dir,
        )
        assert entered.wait(timeout=10)
        try:
            with pytest.raises(
                ReferenceEmbeddingCheckpointBusyError,
                match="writer is busy",
            ):
                build_reference_embeddings(
                    manifest,
                    inputs,
                    scorer=second_scorer,
                    checkpoint_dir=checkpoint_dir,
                )
        finally:
            release.set()
        assert future.result().height == 1

    assert second_scorer.calls == []


def test_build_rejects_tampered_checkpoint_part(tmp_path: Path) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)
    checkpoint_dir = tmp_path / "checkpoint"
    build_reference_embeddings(
        manifest,
        inputs,
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        checkpoint_dir=checkpoint_dir,
    )
    part = next(
        (checkpoint_dir / REFERENCE_EMBEDDINGS_CHECKPOINT_PARTS_DIR).glob("*.parquet")
    )
    part.write_bytes(part.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="part SHA-256 mismatch"):
        build_reference_embeddings(
            manifest,
            inputs,
            scorer=FakeScorer({}),
            checkpoint_dir=checkpoint_dir,
        )


def test_resume_rebinds_checkpoint_provenance_to_frozen_support(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)
    checkpoint_dir = tmp_path / "checkpoint"
    build_reference_embeddings(
        manifest,
        inputs,
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
        checkpoint_dir=checkpoint_dir,
    )
    state_path = checkpoint_dir / REFERENCE_EMBEDDINGS_CHECKPOINT_STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    part_path = (
        checkpoint_dir
        / REFERENCE_EMBEDDINGS_CHECKPOINT_PARTS_DIR
        / state["parts"][0]["file"]
    )
    rows = load_reference_embeddings(part_path).to_dicts()
    rows[0]["accepted_taxon_key"] = "forged-taxon"
    rows[0]["scientific_name"] = "Forged taxon"
    rows[0]["embedding_fingerprint"] = (
        reference_embeddings_module._embedding_row_fingerprint(rows[0])  # noqa: SLF001 - constructs an internally consistent hostile checkpoint.
    )
    forged = pl.DataFrame(
        rows,
        schema=reference_embeddings_schema(3),
        orient="row",
        strict=True,
    )
    reference_embeddings_module.write_parquet(forged, part_path, overwrite=True)
    state["parts"][0]["artifact_fingerprint"] = (
        reference_embeddings_artifact_fingerprint(forged)
    )
    state["parts"][0]["sha256"] = _file_sha256(part_path)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    scorer = FakeScorer({})

    with pytest.raises(ValueError, match="checkpoint provenance mismatch"):
        build_reference_embeddings(
            manifest,
            inputs,
            scorer=scorer,
            embedding_created_at=NOW,
            checkpoint_dir=checkpoint_dir,
        )
    assert scorer.calls == []


def test_build_recovers_when_first_checkpoint_state_commit_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)
    checkpoint_dir = tmp_path / "checkpoint"
    original_write = reference_embeddings_module._write_json_atomically  # noqa: SLF001 - fault injection at the checkpoint commit boundary.

    def fail_state_commit(path: Path, value: dict[str, object]) -> None:
        raise OSError("fixture state commit interruption")

    monkeypatch.setattr(
        reference_embeddings_module,
        "_write_json_atomically",
        fail_state_commit,
    )
    with pytest.raises(OSError, match="state commit interruption"):
        build_reference_embeddings(
            manifest,
            inputs,
            scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
            checkpoint_dir=checkpoint_dir,
        )
    assert not (checkpoint_dir / REFERENCE_EMBEDDINGS_CHECKPOINT_STATE_FILE).exists()

    monkeypatch.setattr(
        reference_embeddings_module,
        "_write_json_atomically",
        original_write,
    )
    scorer = FakeScorer({"media-a.png": [1.0, 0.0, 0.0]})
    result = build_reference_embeddings(
        manifest,
        inputs,
        scorer=scorer,
        checkpoint_dir=checkpoint_dir,
    )

    assert result.height == 1
    assert len(scorer.calls) == 1
    assert (
        len(
            list(
                (checkpoint_dir / REFERENCE_EMBEDDINGS_CHECKPOINT_PARTS_DIR).glob(
                    "*.parquet"
                )
            )
        )
        == 1
    )


def test_build_validates_manifest_and_visual_coverage_before_scorer_work(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(
        tmp_path, (("media-a", "support_train"), ("media-b", "support_train"))
    )
    inputs = _visual_inputs(tmp_path, manifest)
    scorer = FakeScorer({item.image_path.name: [1.0, 0.0, 0.0] for item in inputs})

    with pytest.raises(ValueError, match="missing visual inputs.*media-b"):
        build_reference_embeddings(manifest, inputs[:1], scorer=scorer)
    assert scorer.calls == []

    bad_manifest = manifest.with_columns(
        pl.when(pl.col("reference_media_id") == "media-a")
        .then(pl.lit("sha256:" + "0" * 64))
        .otherwise(pl.col("support_row_fingerprint"))
        .alias("support_row_fingerprint")
    )
    with pytest.raises(ValueError, match="support manifest row fingerprint"):
        build_reference_embeddings(bad_manifest, inputs, scorer=scorer)
    assert scorer.calls == []


def test_build_requires_raw_full_image_input_for_every_eligible_media(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    row = manifest.row(0, named=True)
    source_path = Path(unquote(urlsplit(str(row["source_object_uri"])).path))
    generated = generate_full_frame_attention_variants(
        _decoded_image(source_path),
        (
            AttentionRegion(
                source_detection_id="detection:a",
                route="adult_field",
                bbox_xyxyn=(0.1, 0.1, 0.9, 0.9),
            ),
        ),
        source_type="reference_fixture",
    )
    focused = next(
        variant
        for variant in generated.variants
        if variant.visual_input_kind == FOCUSED_FULL_FRAME_KIND
    )
    focused_path = tmp_path / "focused.png"
    Image.frombytes(
        focused.image.mode,
        (focused.image.width, focused.image.height),
        focused.image.data,
    ).save(focused_path, format="PNG")
    focused_input = ReferenceVisualInput.from_variant(
        reference_media_id="media-a",
        source_image_path=source_path,
        image_path=focused_path,
        variant=focused,
    )
    scorer = FakeScorer({"focused.png": [1.0, 0.0, 0.0]})

    with pytest.raises(ValueError, match="exactly one raw full-image input"):
        build_reference_embeddings(manifest, [focused_input], scorer=scorer)
    assert scorer.calls == []


def test_build_requires_permitting_readiness_and_exact_manifest(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)
    scorer = FakeScorer({"media-a.png": [1.0, 0.0, 0.0]})

    with pytest.raises(ValueError, match="does not authorize"):
        _build_reference_embeddings(
            manifest,
            inputs,
            readiness_permit=replace(_permit(manifest), status="blocked"),
            scorer=scorer,
        )
    with pytest.raises(ValueError, match="support manifest fingerprint mismatch"):
        _build_reference_embeddings(
            manifest,
            inputs,
            readiness_permit=replace(
                _permit(manifest),
                support_manifest_fingerprint=_sha("wrong-manifest"),
            ),
            scorer=scorer,
        )
    with pytest.raises(ValueError, match="visual input contract"):
        _build_reference_embeddings(
            manifest,
            inputs,
            readiness_permit=replace(
                _permit(manifest),
                input_contract_version="different-visual-input-v1",
            ),
            scorer=scorer,
        )
    with pytest.raises(ValueError, match="model revision"):
        _build_reference_embeddings(
            manifest,
            inputs,
            readiness_permit=replace(
                _permit(manifest),
                model_revision="different-revision",
            ),
            scorer=scorer,
        )
    with pytest.raises(ValueError, match="model input fingerprint mismatch"):
        _build_reference_embeddings(
            manifest,
            inputs,
            readiness_permit=replace(
                _permit(manifest),
                model_input_fingerprint=_sha("unrelated-model-input"),
            ),
            scorer=scorer,
        )
    assert scorer.calls == []


@pytest.mark.parametrize("leaking_field", ["observer_id", "duplicate_group_id"])
def test_build_rejects_support_provenance_leaking_across_splits(
    tmp_path: Path,
    leaking_field: str,
) -> None:
    manifest = _support_manifest(
        tmp_path,
        (("media-a", "support_train"), ("media-b", "final_test")),
    )
    rows = manifest.to_dicts()
    rows[1][leaking_field] = rows[0][leaking_field]
    rows[1]["support_row_fingerprint"] = readiness_module._support_row_fingerprint(  # noqa: SLF001 - adversarial fixture rebuilds the persisted row contract.
        rows[1]
    )
    leaking = pl.DataFrame(
        rows,
        schema=reference_support_manifest_schema(),
        orient="row",
    ).sort(
        [
            "accepted_taxon_key",
            "geo_cluster_id",
            "route",
            "support_split",
            "reference_media_id",
        ]
    )
    inputs = _visual_inputs(tmp_path, leaking)
    scorer = FakeScorer({item.image_path.name: [1.0, 0.0, 0.0] for item in inputs})

    with pytest.raises(ValueError, match=f"{leaking_field}="):
        build_reference_embeddings(leaking, inputs, scorer=scorer)
    assert scorer.calls == []


def test_build_rejects_source_path_substitution_even_for_identical_bytes(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)
    scorer = FakeScorer({"media-a.png": [1.0, 0.0, 0.0]})
    substituted = replace(inputs[0], source_image_path=inputs[0].image_path)

    with pytest.raises(ValueError, match="does not match reviewed object URI"):
        build_reference_embeddings(manifest, [substituted], scorer=scorer)
    assert scorer.calls == []


@pytest.mark.parametrize(
    ("config_change", "message"),
    [
        ({"resize_mode": "shortest"}, "resize mode"),
        ({"fill_color": 255}, "fill"),
        ({"size": [336, 336]}, "size"),
        ({"mean": [0.0, 0.0, 0.0]}, "mean"),
        ({"std": [1.0, 1.0, 1.0]}, "std"),
    ],
)
def test_build_rejects_worker_preprocessing_drift(
    tmp_path: Path,
    config_change: dict[str, object],
    message: str,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)
    scorer = FakeScorer({"media-a.png": [1.0, 0.0, 0.0]})
    scorer.preprocessing_config.update(config_change)
    scorer.preprocessing_fingerprint = _preprocessing_attestation_fingerprint(
        scorer.preprocessing_config
    )

    with pytest.raises(ValueError, match=message):
        build_reference_embeddings(manifest, inputs, scorer=scorer)


def test_build_rejects_worker_attestation_not_pinned_by_readiness(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)
    scorer = FakeScorer({"media-a.png": [1.0, 0.0, 0.0]})
    permit = _permit(manifest)
    mismatched_identity = replace(
        permit.model_input_identity(),
        open_clip_version="3.2.0",
    )

    with pytest.raises(ValueError, match="OpenCLIP version does not match"):
        _build_reference_embeddings(
            manifest,
            inputs,
            readiness_permit=replace(
                permit,
                open_clip_version="3.2.0",
                model_input_fingerprint=mismatched_identity.fingerprint,
            ),
            scorer=scorer,
        )

    assert scorer.calls == []


def test_build_rejects_unknown_ineligible_duplicate_and_cross_split_inputs(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(
        tmp_path,
        (("media-a", "support_train"), ("media-b", "final_test")),
        ineligible_ids=("media-x",),
    )
    inputs = _visual_inputs(tmp_path, manifest.filter(pl.col("support_eligible")))
    unknown = replace(inputs[0], reference_media_id="missing")
    with pytest.raises(ValueError, match="unknown or ineligible.*missing"):
        build_reference_embeddings(
            manifest,
            [*inputs, unknown],
            scorer=FakeScorer({}),
        )

    ineligible = replace(inputs[0], reference_media_id="media-x")
    with pytest.raises(ValueError, match="unknown or ineligible.*media-x"):
        build_reference_embeddings(
            manifest,
            [*inputs, ineligible],
            scorer=FakeScorer({}),
        )

    with pytest.raises(ValueError, match="duplicate reference visual input"):
        build_reference_embeddings(
            manifest,
            [*inputs, inputs[0]],
            scorer=FakeScorer({}),
        )

    same_checkpoint_key = replace(
        inputs[0],
        transformation_version="different-version-with-same-identity",
    )
    with pytest.raises(ValueError, match="checkpoint identity"):
        build_reference_embeddings(
            manifest,
            [*inputs, same_checkpoint_key],
            scorer=FakeScorer({}),
        )

    duplicate_pixels = _image(tmp_path / "duplicate.png", (1, 2, 3))
    first = replace(
        inputs[0],
        reference_media_id="media-a",
        image_path=duplicate_pixels,
        image_content_hash=decoded_image_file_content_hash(duplicate_pixels),
    )
    second = replace(first, reference_media_id="media-b")
    with pytest.raises(ValueError, match="image content crosses support splits"):
        build_reference_embeddings(
            manifest,
            [first, second],
            scorer=FakeScorer({}),
        )


def test_build_rejects_visual_hash_mismatch_before_model_load(tmp_path: Path) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)
    scorer = FakeScorer({inputs[0].image_path.name: [1.0, 0.0, 0.0]})
    changed = _image(inputs[0].image_path, (200, 100, 50))
    assert changed == inputs[0].image_path

    with pytest.raises(ValueError, match="decoded content hash mismatch"):
        build_reference_embeddings(manifest, inputs, scorer=scorer)

    assert scorer.calls == []


def test_build_rejects_file_replacement_at_worker_decode_boundary(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)

    class ReplacingScorer(FakeScorer):
        def embed_image_paths(self, image_paths: list[Path]) -> list[list[float]]:
            _image(image_paths[0], (200, 100, 50))
            return super().embed_image_paths(image_paths)

    scorer = ReplacingScorer({"media-a.png": [1.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="embedded content does not match"):
        build_reference_embeddings(manifest, inputs, scorer=scorer)

    assert len(scorer.calls) == 1


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ({"media-a.png": []}, "non-empty"),
        ({"media-a.png": [0.0, 0.0, 0.0]}, "non-zero norm"),
        ({"media-a.png": [float("nan"), 0.0, 1.0]}, "finite values"),
        ({"media-a.png": [2.0, 0.0, 0.0]}, "unit-normalized"),
    ],
)
def test_build_rejects_invalid_worker_vectors(
    tmp_path: Path,
    vectors: dict[str, list[float]],
    message: str,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)

    with pytest.raises(ValueError, match=message):
        build_reference_embeddings(
            manifest,
            inputs,
            scorer=FakeScorer(vectors),
        )


def test_build_rejects_mixed_dimensions_and_result_count(tmp_path: Path) -> None:
    manifest = _support_manifest(
        tmp_path, (("media-a", "support_train"), ("media-b", "support_train"))
    )
    inputs = _visual_inputs(tmp_path, manifest)
    mixed = FakeScorer({"media-a.png": [1.0, 0.0], "media-b.png": [1.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="mixed embedding dimensions"):
        build_reference_embeddings(manifest, inputs, scorer=mixed)

    class ShortScorer(FakeScorer):
        def embed_image_paths(self, image_paths: list[Path]) -> list[list[float]]:
            self.calls.append(tuple(image_paths))
            self.last_image_content_hashes = tuple(
                decoded_image_file_content_hash(path) for path in image_paths
            )
            return [[1.0, 0.0]]

    with pytest.raises(ValueError, match="returned 1 rows for 2 images"):
        build_reference_embeddings(
            manifest,
            inputs,
            scorer=ShortScorer({}),
            batch_size=2,
        )


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("image_resize_mode", "shortest", "longest-side resize"),
        ("model_id", "", "model_id"),
        ("model_revision", "", "model_revision"),
        ("model_weights_sha256", None, "model weights SHA-256"),
    ],
)
def test_build_rejects_unfrozen_scorer_identity(
    tmp_path: Path,
    attribute: str,
    value: object,
    message: str,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)
    scorer = FakeScorer({"media-a.png": [1.0, 0.0, 0.0]})
    setattr(scorer, attribute, value)

    with pytest.raises(ValueError, match=message):
        build_reference_embeddings(manifest, inputs, scorer=scorer)


def test_publish_and_load_round_trip_fixed_size_embeddings(tmp_path: Path) -> None:
    manifest = _support_manifest(
        tmp_path, (("media-a", "support_train"), ("media-b", "final_test"))
    )
    inputs = _visual_inputs(tmp_path, manifest)
    frame = build_reference_embeddings(
        manifest,
        inputs,
        scorer=FakeScorer(
            {"media-a.png": [1.0, 0.0, 0.0], "media-b.png": [0.0, 1.0, 0.0]}
        ),
        embedding_created_at=NOW,
    )

    paths = publish_reference_embeddings(
        frame,
        tmp_path / "published",
        run_id="reference-embedding-test",
    )
    loaded = load_reference_embeddings(paths["embeddings"])
    loaded_from_directory = load_reference_embeddings(tmp_path / "published")

    assert set(paths) == {"embeddings", "report", "summary", "manifest"}
    assert paths["embeddings"].name == REFERENCE_EMBEDDINGS_FILE
    assert paths["report"].name == REFERENCE_EMBEDDINGS_REPORT_FILE
    assert paths["summary"].name == REFERENCE_EMBEDDINGS_SUMMARY_FILE
    assert paths["manifest"].name == REFERENCE_EMBEDDINGS_MANIFEST_FILE
    assert_frame_equal(frame, loaded, check_exact=True)
    assert_frame_equal(frame, loaded_from_directory, check_exact=True)
    arrow_field = pq.read_schema(paths["embeddings"]).field("embedding")
    assert pa.types.is_fixed_size_list(arrow_field.type)
    assert arrow_field.type.list_size == 3
    assert b"ARROW:schema" in pq.ParquetFile(paths["embeddings"]).metadata.metadata
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["run_id"] == "reference-embedding-test"
    assert report["row_count"] == 2
    assert report["embedding_dimension"] == 3
    assert report["support_split_counts"] == {"final_test": 1, "support_train": 1}
    assert report["artifact"]["sha256"] == _file_sha256(paths["embeddings"])
    assert report["artifact_fingerprint"] == (
        reference_embeddings_artifact_fingerprint(frame)
    )
    assert "# Reference embedding build" in paths["summary"].read_text(encoding="utf-8")
    published_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert published_manifest["status"] == "complete"
    assert published_manifest["artifact_set_version"] == (
        reference_embeddings_artifact_fingerprint(frame)
    )
    records = {item["name"]: item for item in published_manifest["files"]}
    for name, key in (
        (REFERENCE_EMBEDDINGS_FILE, "embeddings"),
        (REFERENCE_EMBEDDINGS_REPORT_FILE, "report"),
        (REFERENCE_EMBEDDINGS_SUMMARY_FILE, "summary"),
    ):
        assert records[name]["uri"] == paths[key].resolve().as_uri()
        assert records[name]["sha256"] == _file_sha256(paths[key])

    with pytest.raises(FileExistsError):
        publish_reference_embeddings(frame, tmp_path / "published")


def test_local_publication_reader_rejects_missing_or_tampered_commit_set(
    tmp_path: Path,
) -> None:
    support = _support_manifest(tmp_path, (("media-a", "support_train"),))
    frame = build_reference_embeddings(
        support,
        _visual_inputs(tmp_path, support),
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )
    directory = tmp_path / "published"
    paths = publish_reference_embeddings(frame, directory, run_id="local-gate")
    manifest_bytes = paths["manifest"].read_bytes()
    report_bytes = paths["report"].read_bytes()
    summary_bytes = paths["summary"].read_bytes()

    paths["manifest"].unlink()
    with pytest.raises(ValueError, match="publication manifest is missing"):
        load_reference_embeddings(directory)
    with pytest.raises(ValueError, match="publication manifest is missing"):
        load_reference_embeddings(paths["embeddings"])
    paths["manifest"].write_bytes(manifest_bytes)

    manifest = json.loads(manifest_bytes)
    manifest["files"][0]["sha256"] = _sha("forged-embeddings")
    paths["manifest"].write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest metadata mismatch"):
        load_reference_embeddings(directory)
    paths["manifest"].write_bytes(manifest_bytes)

    report = json.loads(report_bytes)
    report["row_count"] = 99
    paths["report"].write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="report identity mismatch"):
        load_reference_embeddings(directory)
    paths["report"].write_bytes(report_bytes)

    paths["summary"].write_text(
        summary_bytes.decode("utf-8") + "tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="summary/report mismatch"):
        load_reference_embeddings(directory)
    paths["summary"].write_bytes(summary_bytes)

    assert_frame_equal(frame, load_reference_embeddings(directory), check_exact=True)


def test_non_default_preprocessing_contract_build_publish_load_round_trip(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)
    contract = TargetPreprocessingContract(
        version="target-full-frame-openclip-preprocess-test-v2",
        image_size_px=238,
        normalization_mean=(0.5, 0.4, 0.3),
        normalization_std=(0.2, 0.25, 0.3),
    )
    scorer = FakeScorer({"media-a.png": [1.0, 0.0, 0.0]})
    scorer.preprocessing_config.update(
        {
            "mean": list(contract.normalization_mean),
            "size": [contract.image_size_px, contract.image_size_px],
            "std": list(contract.normalization_std),
        }
    )
    scorer.preprocessing_fingerprint = _preprocessing_attestation_fingerprint(
        scorer.preprocessing_config
    )
    default_permit = _permit(manifest)
    identity = ReferenceModelInputIdentity(
        schema_version=default_permit.model_input_schema_version,
        model_name=default_permit.model_name,
        model_version=default_permit.model_version,
        model_revision=default_permit.model_revision,
        checkpoint_uri=default_permit.checkpoint_uri,
        checkpoint_sha256=default_permit.checkpoint_sha256,
        open_clip_version=default_permit.open_clip_version,
        open_clip_config_sha256=default_permit.open_clip_config_sha256,
        preprocessing_version=contract.version,
        preprocessing_contract_fingerprint=contract.fingerprint,
        preprocessing_attestation_fingerprint=scorer.preprocessing_fingerprint,
        input_contract_version=default_permit.input_contract_version,
    )
    permit = replace(
        default_permit,
        preprocessing_version=contract.version,
        preprocessing_contract_fingerprint=contract.fingerprint,
        preprocessing_attestation_fingerprint=scorer.preprocessing_fingerprint,
        model_input_fingerprint=identity.fingerprint,
    )

    frame = _build_reference_embeddings(
        manifest,
        inputs,
        readiness_permit=permit,
        scorer=scorer,
        preprocessing_contract=contract,
        embedding_created_at=NOW,
    )
    paths = publish_reference_embeddings(frame, tmp_path / "published-custom")
    loaded = load_reference_embeddings(paths["embeddings"])

    assert loaded["preprocessing_version"].unique().to_list() == [contract.version]
    assert loaded["preprocessing_fingerprint"].unique().to_list() == [
        contract.fingerprint
    ]
    assert json.loads(loaded["preprocessing_config_json"][0]) == (
        scorer.preprocessing_config
    )
    assert_frame_equal(frame, loaded, check_exact=True)


def test_cloud_publication_is_manifest_committed_immutable_and_run_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)
    frame = build_reference_embeddings(
        manifest,
        inputs,
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )
    storage = LocalStorageBackend()
    workstore = SQLiteWorkStore(tmp_path / "work.sqlite3")
    first_prefix = (tmp_path / "cloud" / "run-1").resolve().as_uri()
    monkeypatch.chdir(tmp_path)

    first = publish_reference_embeddings_to_cloud(
        frame,
        first_prefix,
        storage=storage,
        workstore=workstore,
        job_name="reference-embeddings",
        stage="frozen-support",
        registry_version="butterflies-v1",
        run_id="run-1",
        worker_id="worker-1",
    )
    first_report = storage.read_text(first["report"])
    first_manifest = storage.read_text(first["manifest"])
    committed = json.loads(first_manifest)

    assert set(first) == {"embeddings", "report", "summary", "manifest"}
    assert committed["schema_version"] == "biominer-artifact-manifest-v1"
    assert committed["status"] == "complete"
    assert committed["job_name"] == "reference-embeddings"
    assert committed["stage"] == "frozen-support"
    assert committed["git_sha"] == current_git_sha()
    assert [item["name"] for item in committed["files"]] == [
        REFERENCE_EMBEDDINGS_FILE,
        REFERENCE_EMBEDDINGS_REPORT_FILE,
        REFERENCE_EMBEDDINGS_SUMMARY_FILE,
    ]

    retried = publish_reference_embeddings_to_cloud(
        frame,
        first_prefix,
        storage=storage,
        workstore=workstore,
        job_name="reference-embeddings",
        stage="frozen-support",
        registry_version="butterflies-v1",
        run_id="run-1",
        worker_id="worker-1",
    )
    assert retried == first
    assert storage.read_text(first["report"]) == first_report
    assert storage.read_text(first["manifest"]) == first_manifest

    with pytest.raises(ValueError, match="manifest identity mismatch: job_name"):
        publish_reference_embeddings_to_cloud(
            frame,
            first_prefix,
            storage=storage,
            workstore=workstore,
            job_name="different-job",
            stage="frozen-support",
            registry_version="butterflies-v1",
            run_id="run-1",
            worker_id="worker-1",
        )
    assert not workstore.list_committed_shards(
        job_name="different-job",
        stage="frozen-support",
        registry_version="butterflies-v1",
        run_id="run-1",
    )

    second = publish_reference_embeddings_to_cloud(
        frame,
        (tmp_path / "cloud" / "run-2").resolve().as_uri(),
        storage=storage,
        workstore=workstore,
        job_name="reference-embeddings",
        stage="frozen-support",
        registry_version="butterflies-v1",
        run_id="run-2",
        worker_id="worker-2",
    )
    assert second["embeddings"] != first["embeddings"]
    second_committed = json.loads(storage.read_text(second["manifest"]))
    first_files = {item["name"]: item for item in committed["files"]}
    second_files = {item["name"]: item for item in second_committed["files"]}
    for name in (
        REFERENCE_EMBEDDINGS_REPORT_FILE,
        REFERENCE_EMBEDDINGS_SUMMARY_FILE,
    ):
        assert (
            first_files[name]["semantic_fingerprint"]
            == second_files[name]["semantic_fingerprint"]
        )
        assert first_files[name]["sha256"] != second_files[name]["sha256"]
    assert (
        committed["dependency_fingerprints"]["model_fingerprint"]
        == frame["model_fingerprint"][0]
    )
    assert (
        len(
            workstore.list_committed_shards(
                job_name="reference-embeddings",
                stage="frozen-support",
                registry_version="butterflies-v1",
                run_id="run-1",
            )
        )
        == 1
    )
    assert (
        len(
            workstore.list_committed_shards(
                job_name="reference-embeddings",
                stage="frozen-support",
                registry_version="butterflies-v1",
                run_id="run-2",
            )
        )
        == 1
    )

    first_embedding_path = Path(unquote(urlsplit(first["embeddings"]).path))
    first_embedding_path.write_bytes(first_embedding_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="size mismatch|checksum mismatch"):
        publish_reference_embeddings_to_cloud(
            frame,
            first_prefix,
            storage=storage,
            workstore=workstore,
            job_name="reference-embeddings",
            stage="frozen-support",
            registry_version="butterflies-v1",
            run_id="run-1",
            worker_id="worker-1",
        )


def test_cloud_publication_serializes_identical_concurrent_publishers(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    frame = build_reference_embeddings(
        manifest,
        _visual_inputs(tmp_path, manifest),
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )
    storage = LocalStorageBackend()
    workstore = SQLiteWorkStore(tmp_path / "work.sqlite3")
    prefix = (tmp_path / "cloud" / "shared-run").resolve().as_uri()

    def publish(worker_id: str) -> dict[str, str]:
        return publish_reference_embeddings_to_cloud(
            frame,
            prefix,
            storage=storage,
            workstore=workstore,
            job_name="reference-embeddings",
            stage="frozen-support",
            registry_version="butterflies-v1",
            run_id="shared-run",
            worker_id=worker_id,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, ("worker-1", "worker-2")))

    assert results[0] == results[1]
    assert (
        len(
            workstore.list_committed_shards(
                job_name="reference-embeddings",
                stage="frozen-support",
                registry_version="butterflies-v1",
                run_id="shared-run",
            )
        )
        == 1
    )


def test_cloud_publication_retries_after_report_write_failure(
    tmp_path: Path,
) -> None:
    class FailFirstReportWrite(LocalStorageBackend):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def write_file(
            self,
            uri,
            source,
            *,
            content_type=None,
            overwrite=True,
        ) -> str:
            if str(uri).endswith(REFERENCE_EMBEDDINGS_REPORT_FILE) and not self.failed:
                self.failed = True
                raise OSError("fixture report write failure")
            return super().write_file(
                uri,
                source,
                content_type=content_type,
                overwrite=overwrite,
            )

    support = _support_manifest(tmp_path, (("media-a", "support_train"),))
    frame = build_reference_embeddings(
        support,
        _visual_inputs(tmp_path, support),
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )
    storage = FailFirstReportWrite()
    workstore = SQLiteWorkStore(tmp_path / "work.sqlite3")
    prefix = (tmp_path / "cloud" / "retry-report").resolve().as_uri()

    with pytest.raises(OSError, match="report write failure"):
        publish_reference_embeddings_to_cloud(
            frame,
            prefix,
            storage=storage,
            workstore=workstore,
            job_name="reference-embeddings",
            stage="frozen-support",
            registry_version="butterflies-v1",
            run_id="run-1",
            worker_id="worker-1",
        )
    assert storage.exists(join_uri(prefix, REFERENCE_EMBEDDINGS_FILE))
    assert not storage.exists(join_uri(prefix, REFERENCE_EMBEDDINGS_REPORT_FILE))
    assert not storage.exists(join_uri(prefix, REFERENCE_EMBEDDINGS_SUMMARY_FILE))
    assert not storage.exists(join_uri(prefix, "manifest.json"))

    retry_frame = build_reference_embeddings(
        support,
        _visual_inputs(tmp_path, support),
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW + timedelta(days=1),
    ).with_columns(pl.lit(_sha("retry-readiness-object")).alias("readiness_sha256"))
    paths = publish_reference_embeddings_to_cloud(
        retry_frame,
        prefix,
        storage=storage,
        workstore=workstore,
        job_name="reference-embeddings",
        stage="frozen-support",
        registry_version="butterflies-v1",
        run_id="run-1",
        worker_id="worker-1",
    )
    assert storage.exists(paths["manifest"])
    persisted = load_reference_embeddings(
        Path(unquote(urlsplit(paths["embeddings"]).path))
    )
    assert persisted["embedding_created_at"].to_list() == [NOW]
    assert persisted["readiness_sha256"].to_list() == [_sha("readiness")]
    report = storage.read_json(paths["report"])
    manifest = storage.read_json(paths["manifest"])
    assert report["readiness_sha256"] == _sha("readiness")
    assert manifest["dependency_fingerprints"]["readiness_sha256"] == _sha("readiness")


@pytest.mark.parametrize(
    "tamper",
    [
        "effective_configuration",
        "qa",
        "metrics",
        "embedding_primary_key",
        "embedding_sort_order",
        "report_semantic_fingerprint",
    ],
)
def test_cloud_manifest_rejects_corrupt_authoritative_metadata(
    tmp_path: Path,
    tamper: str,
) -> None:
    support = _support_manifest(tmp_path, (("media-a", "support_train"),))
    frame = build_reference_embeddings(
        support,
        _visual_inputs(tmp_path, support),
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )
    storage = LocalStorageBackend()
    workstore = SQLiteWorkStore(tmp_path / "work.sqlite3")
    prefix = (tmp_path / "cloud" / tamper).resolve().as_uri()
    paths = publish_reference_embeddings_to_cloud(
        frame,
        prefix,
        storage=storage,
        workstore=workstore,
        job_name="reference-embeddings",
        stage="frozen-support",
        registry_version="butterflies-v1",
        run_id="run-1",
        worker_id="worker-1",
    )
    manifest_path = Path(unquote(urlsplit(paths["manifest"]).path))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tamper == "effective_configuration":
        payload["effective_configuration"]["embedding_dimension"] = 999
    elif tamper == "qa":
        payload["qa"] = {"status": "failed", "fatal_count": 99, "warning_count": 0}
    elif tamper == "metrics":
        payload["metrics"]["row_count"] = 999
    else:
        records = {item["name"]: item for item in payload["files"]}
        if tamper == "embedding_primary_key":
            records[REFERENCE_EMBEDDINGS_FILE]["primary_key"] = ["reference_media_id"]
        elif tamper == "embedding_sort_order":
            records[REFERENCE_EMBEDDINGS_FILE]["sort_order"] = []
        else:
            records[REFERENCE_EMBEDDINGS_REPORT_FILE]["semantic_fingerprint"] = _sha(
                "forged-report-semantic-fingerprint"
            )
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="authoritative metadata mismatch"):
        publish_reference_embeddings_to_cloud(
            frame,
            prefix,
            storage=storage,
            workstore=workstore,
            job_name="reference-embeddings",
            stage="frozen-support",
            registry_version="butterflies-v1",
            run_id="run-1",
            worker_id="worker-1",
        )


def test_cloud_manifest_rejects_rehashed_semantically_corrupt_report(
    tmp_path: Path,
) -> None:
    support = _support_manifest(tmp_path, (("media-a", "support_train"),))
    frame = build_reference_embeddings(
        support,
        _visual_inputs(tmp_path, support),
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )
    storage = LocalStorageBackend()
    workstore = SQLiteWorkStore(tmp_path / "work.sqlite3")
    prefix = (tmp_path / "cloud" / "corrupt-report").resolve().as_uri()
    paths = publish_reference_embeddings_to_cloud(
        frame,
        prefix,
        storage=storage,
        workstore=workstore,
        job_name="reference-embeddings",
        stage="frozen-support",
        registry_version="butterflies-v1",
        run_id="run-1",
        worker_id="worker-1",
    )
    report_path = Path(unquote(urlsplit(paths["report"]).path))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "failed"
    report_path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_path = Path(unquote(urlsplit(paths["manifest"]).path))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_record = next(
        item
        for item in manifest_payload["files"]
        if item["name"] == REFERENCE_EMBEDDINGS_REPORT_FILE
    )
    report_record["byte_count"] = report_path.stat().st_size
    report_record["sha256"] = _file_sha256(report_path)
    report_record["semantic_fingerprint"] = (
        reference_embeddings_module._json_fingerprint(report)  # noqa: SLF001 - models a rehashed corrupt artifact set.
    )
    manifest_path.write_text(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cloud report identity mismatch"):
        publish_reference_embeddings_to_cloud(
            frame,
            prefix,
            storage=storage,
            workstore=workstore,
            job_name="reference-embeddings",
            stage="frozen-support",
            registry_version="butterflies-v1",
            run_id="run-1",
            worker_id="worker-1",
        )


def test_validator_rejects_schema_vector_norm_and_fingerprint_tampering(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    inputs = _visual_inputs(tmp_path, manifest)
    frame = build_reference_embeddings(
        manifest,
        inputs,
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )

    with pytest.raises(ValueError, match="physical schema"):
        validate_reference_embeddings(
            frame.with_columns(pl.col("embedding").cast(pl.List(pl.Float32)))
        )
    with pytest.raises(ValueError, match="embedding norm mismatch"):
        validate_reference_embeddings(
            frame.with_columns(pl.lit(0.5).alias("embedding_norm"))
        )
    with pytest.raises(ValueError, match="embedding fingerprint mismatch"):
        validate_reference_embeddings(
            frame.with_columns(
                pl.lit("sha256:" + "0" * 64).alias("embedding_fingerprint")
            )
        )
    rows = frame.to_dicts()
    decision_id = rows[0]["review_decision_ids"][0]
    rows[0]["review_decision_ids"] = [decision_id, decision_id]
    rows[0]["embedding_fingerprint"] = (
        reference_embeddings_module._embedding_row_fingerprint(rows[0])  # noqa: SLF001 - adversarial persisted-row fixture.
    )
    duplicated_decisions = pl.DataFrame(
        rows,
        schema=reference_embeddings_schema(3),
        orient="row",
        strict=True,
    )
    with pytest.raises(ValueError, match="review_decision_ids.*sorted and unique"):
        validate_reference_embeddings(duplicated_decisions)


def test_validator_binds_model_id_and_embedding_output_contract(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    frame = build_reference_embeddings(
        manifest,
        _visual_inputs(tmp_path, manifest),
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )
    row = frame.row(0, named=True)

    assert row["model_fingerprint"] != row["model_input_fingerprint"]
    assert row["model_fingerprint"] == (
        reference_embeddings_module._reference_embedding_model_fingerprint(  # noqa: SLF001 - verifies the persisted model dependency contract.
            model_input_fingerprint=row["model_input_fingerprint"],
            embedding_dimension=3,
        )
    )

    rows = frame.to_dicts()
    rows[0]["model_id"] = "different/model"
    rows[0]["embedding_fingerprint"] = (
        reference_embeddings_module._embedding_row_fingerprint(rows[0])  # noqa: SLF001 - adversarial persisted-row fixture.
    )
    mismatched_id = pl.DataFrame(
        rows,
        schema=reference_embeddings_schema(3),
        orient="row",
        strict=True,
    )
    with pytest.raises(ValueError, match="model ID does not match model name"):
        validate_reference_embeddings(mismatched_id)

    rows = frame.to_dicts()
    rows[0]["model_fingerprint"] = rows[0]["model_input_fingerprint"]
    rows[0]["embedding_fingerprint"] = (
        reference_embeddings_module._embedding_row_fingerprint(rows[0])  # noqa: SLF001 - adversarial persisted-row fixture.
    )
    mismatched_output = pl.DataFrame(
        rows,
        schema=reference_embeddings_schema(3),
        orient="row",
        strict=True,
    )
    with pytest.raises(ValueError, match="model_fingerprint mismatch"):
        validate_reference_embeddings(mismatched_output)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"route": "unknown-route"}, "unsupported reference route"),
        ({"life_stage": "larva"}, "route dimensions mismatch"),
        ({"visual_domain": "pinned_specimen"}, "route dimensions mismatch"),
    ],
)
def test_validator_rejects_invalid_or_inconsistent_route_dimensions(
    tmp_path: Path,
    changes: dict[str, str],
    message: str,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    frame = build_reference_embeddings(
        manifest,
        _visual_inputs(tmp_path, manifest),
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )
    rows = frame.to_dicts()
    rows[0].update(changes)
    rows[0]["embedding_fingerprint"] = (
        reference_embeddings_module._embedding_row_fingerprint(rows[0])  # noqa: SLF001 - adversarial persisted-row fixture.
    )
    tampered = pl.DataFrame(
        rows,
        schema=reference_embeddings_schema(3),
        orient="row",
        strict=True,
    )

    with pytest.raises(ValueError, match=message):
        validate_reference_embeddings(tampered)


def test_validator_rejects_self_consistent_non_target_preprocessing(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    frame = build_reference_embeddings(
        manifest,
        _visual_inputs(tmp_path, manifest),
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )
    rows = frame.to_dicts()
    row = rows[0]
    config = dict(PREPROCESSING_CONFIG)
    config["resize_mode"] = "shortest"
    row["preprocessing_config_json"] = json.dumps(
        config,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    row["preprocessing_attestation_fingerprint"] = (
        reference_embeddings_module._preprocessing_attestation_fingerprint(  # noqa: SLF001 - constructs an internally consistent hostile attestation.
            open_clip_version=row["open_clip_version"],
            open_clip_config_sha256=row["open_clip_config_sha256"],
            preprocessing_version=row["preprocessing_attestation_version"],
            preprocessing_config=config,
        )
    )
    row["model_input_fingerprint"] = ReferenceModelInputIdentity(
        schema_version=row["model_input_schema_version"],
        model_name=row["model_name"],
        model_version=row["model_version"],
        model_revision=row["model_revision"],
        checkpoint_uri=row["model_checkpoint_uri"],
        checkpoint_sha256=row["model_weights_sha256"],
        open_clip_version=row["open_clip_version"],
        open_clip_config_sha256=row["open_clip_config_sha256"],
        preprocessing_version=row["preprocessing_version"],
        preprocessing_contract_fingerprint=row["preprocessing_fingerprint"],
        preprocessing_attestation_fingerprint=row[
            "preprocessing_attestation_fingerprint"
        ],
        input_contract_version=row["input_contract_version"],
    ).fingerprint
    row["model_fingerprint"] = (
        reference_embeddings_module._reference_embedding_model_fingerprint(  # noqa: SLF001 - completes the hostile dependency chain.
            model_input_fingerprint=row["model_input_fingerprint"],
            embedding_dimension=row["embedding_dimension"],
        )
    )
    row["embedding_fingerprint"] = (
        reference_embeddings_module._embedding_row_fingerprint(row)  # noqa: SLF001 - completes the hostile row fingerprint.
    )
    tampered = pl.DataFrame(
        rows,
        schema=reference_embeddings_schema(3),
        orient="row",
        strict=True,
    )

    with pytest.raises(ValueError, match="resize mode does not match target contract"):
        validate_reference_embeddings(tampered)


def test_validator_requires_exactly_one_raw_full_image_row_per_media(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    frame = build_reference_embeddings(
        manifest,
        _visual_inputs(tmp_path, manifest),
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )
    raw = frame.row(0, named=True)
    focused = dict(raw)
    focused["visual_input_kind"] = FOCUSED_FULL_FRAME_KIND
    focused["transformation_version"] = "focused-test-v1"
    focused["transformation_policy_fingerprint"] = _sha("focused-policy")
    focused["transformation_fingerprint"] = _sha("focused-transform")
    focused["visual_input_id"] = (
        reference_embeddings_module._expected_visual_input_id_from_values(  # noqa: SLF001 - creates a structurally valid persisted visual input.
            visual_input_kind=FOCUSED_FULL_FRAME_KIND,
            raw_image_content_hash=focused["raw_image_content_hash"],
            image_content_hash=focused["image_content_hash"],
            transformation_fingerprint=focused["transformation_fingerprint"],
        )
    )
    focused["embedding_fingerprint"] = (
        reference_embeddings_module._embedding_row_fingerprint(focused)  # noqa: SLF001 - completes the persisted row fixture.
    )
    focused_only = pl.DataFrame(
        [focused],
        schema=reference_embeddings_schema(3),
        orient="row",
        strict=True,
    )
    with pytest.raises(ValueError, match="exactly one raw full-image row"):
        validate_reference_embeddings(focused_only)

    complete = reference_embeddings_module._sort_embedding_frame(  # noqa: SLF001 - preserve required physical ordering.
        pl.DataFrame(
            [raw, focused],
            schema=reference_embeddings_schema(3),
            orient="row",
            strict=True,
        )
    )
    validate_reference_embeddings(complete)

    forged_projection = dict(focused)
    forged_projection["accepted_taxon_key"] = "gbif:forged"
    forged_projection["embedding_fingerprint"] = (
        reference_embeddings_module._embedding_row_fingerprint(  # noqa: SLF001 - creates an internally consistent hostile row.
            forged_projection
        )
    )
    mismatched_projection = reference_embeddings_module._sort_embedding_frame(  # noqa: SLF001 - preserve required physical ordering.
        pl.DataFrame(
            [raw, forged_projection],
            schema=reference_embeddings_schema(3),
            orient="row",
            strict=True,
        )
    )
    with pytest.raises(ValueError, match="support projection mismatch"):
        validate_reference_embeddings(mismatched_projection)

    conflicting_vector = dict(focused)
    conflicting_vector["embedding"] = [0.0, 1.0, 0.0]
    conflicting_vector["embedding_norm"] = 1.0
    conflicting_vector["embedding_fingerprint"] = (
        reference_embeddings_module._embedding_row_fingerprint(  # noqa: SLF001 - creates an internally consistent cache conflict.
            conflicting_vector
        )
    )
    conflicting_cache = reference_embeddings_module._sort_embedding_frame(  # noqa: SLF001 - preserve required physical ordering.
        pl.DataFrame(
            [raw, conflicting_vector],
            schema=reference_embeddings_schema(3),
            orient="row",
            strict=True,
        )
    )
    with pytest.raises(ValueError, match="conflicting vectors"):
        validate_reference_embeddings(conflicting_cache)

    second_raw = dict(raw)
    second_raw["raw_image_content_hash"] = _sha("different-raw-content")
    second_raw["image_content_hash"] = second_raw["raw_image_content_hash"]
    second_raw["visual_input_id"] = (
        reference_embeddings_module._expected_visual_input_id_from_values(  # noqa: SLF001 - creates a structurally valid second raw identity.
            visual_input_kind=RAW_FULL_IMAGE_KIND,
            raw_image_content_hash=second_raw["raw_image_content_hash"],
            image_content_hash=second_raw["image_content_hash"],
            transformation_fingerprint=second_raw["transformation_fingerprint"],
        )
    )
    second_raw["embedding_fingerprint"] = (
        reference_embeddings_module._embedding_row_fingerprint(second_raw)  # noqa: SLF001 - completes the persisted row fixture.
    )
    duplicate_raw = reference_embeddings_module._sort_embedding_frame(  # noqa: SLF001 - preserve required physical ordering.
        pl.DataFrame(
            [raw, second_raw],
            schema=reference_embeddings_schema(3),
            orient="row",
            strict=True,
        )
    )
    with pytest.raises(ValueError, match="exactly one raw full-image row"):
        validate_reference_embeddings(duplicate_raw)


def test_validator_rejects_unknown_reference_view(tmp_path: Path) -> None:
    manifest = _support_manifest(tmp_path, (("media-a", "support_train"),))
    frame = build_reference_embeddings(
        manifest,
        _visual_inputs(tmp_path, manifest),
        scorer=FakeScorer({"media-a.png": [1.0, 0.0, 0.0]}),
        embedding_created_at=NOW,
    )
    row = frame.row(0, named=True)
    row["view"] = "nonsense"
    row["embedding_fingerprint"] = (
        reference_embeddings_module._embedding_row_fingerprint(row)  # noqa: SLF001 - creates an internally consistent hostile row.
    )
    tampered = pl.DataFrame(
        [row],
        schema=reference_embeddings_schema(3),
        orient="row",
        strict=True,
    )

    with pytest.raises(ValueError, match="unsupported reference view"):
        validate_reference_embeddings(tampered)


@pytest.mark.parametrize(
    "leaking_field",
    ["reference_media_id", "reference_observation_id", "duplicate_group_id"],
)
def test_validator_rejects_persisted_provenance_crossing_support_splits(
    tmp_path: Path,
    leaking_field: str,
) -> None:
    manifest = _support_manifest(
        tmp_path,
        (("media-a", "support_train"), ("media-b", "final_test")),
    )
    frame = build_reference_embeddings(
        manifest,
        _visual_inputs(tmp_path, manifest),
        scorer=FakeScorer(
            {"media-a.png": [1.0, 0.0, 0.0], "media-b.png": [0.0, 1.0, 0.0]}
        ),
        embedding_created_at=NOW,
    )
    rows = frame.to_dicts()
    rows[1][leaking_field] = rows[0][leaking_field]
    rows[1]["embedding_fingerprint"] = (
        reference_embeddings_module._embedding_row_fingerprint(rows[1])  # noqa: SLF001 - adversarial persisted-row fixture.
    )
    leaking = reference_embeddings_module._sort_embedding_frame(  # noqa: SLF001 - preserve the required physical sort before validation.
        pl.DataFrame(
            rows,
            schema=reference_embeddings_schema(3),
            orient="row",
            strict=True,
        )
    )

    with pytest.raises(ValueError, match=f"reference {leaking_field} crosses"):
        validate_reference_embeddings(leaking)


def test_validator_rejects_duplicate_authoritative_checkpoint_keys(
    tmp_path: Path,
) -> None:
    manifest = _support_manifest(
        tmp_path,
        (("media-a", "support_train"), ("media-b", "support_train")),
    )
    inputs = _visual_inputs(tmp_path, manifest)
    frame = build_reference_embeddings(
        manifest,
        inputs,
        scorer=FakeScorer(
            {"media-a.png": [1.0, 0.0, 0.0], "media-b.png": [0.0, 1.0, 0.0]}
        ),
        embedding_created_at=NOW,
    )
    rows = frame.to_dicts()
    rows[1]["support_row_fingerprint"] = rows[0]["support_row_fingerprint"]
    rows[1]["visual_input_id"] = rows[0]["visual_input_id"]
    rows[1]["embedding_fingerprint"] = (
        reference_embeddings_module._embedding_row_fingerprint(rows[1])  # noqa: SLF001 - adversarial persisted-row fixture.
    )
    tampered = reference_embeddings_module._sort_embedding_frame(  # noqa: SLF001 - preserve the required physical sort before validation.
        pl.DataFrame(
            rows,
            schema=reference_embeddings_schema(3),
            orient="row",
            strict=True,
        )
    )

    with pytest.raises(ValueError, match="duplicate checkpoint keys"):
        validate_reference_embeddings(tampered)


def _support_manifest(
    tmp_path: Path,
    eligible: tuple[tuple[str, str], ...],
    *,
    ineligible_ids: tuple[str, ...] = (),
) -> pl.DataFrame:
    rows = [
        _support_row(
            media_id,
            source_path=_image(
                tmp_path / "source" / f"{media_id}.png",
                (index, index + 1, index + 2),
            ),
            support_split=support_split,
            eligible=True,
        )
        for index, (media_id, support_split) in enumerate(eligible, start=1)
    ]
    rows.extend(
        _support_row(
            media_id,
            source_path=_image(
                tmp_path / "source" / f"{media_id}.png",
                (index + 100, index + 101, index + 102),
            ),
            support_split=None,
            eligible=False,
        )
        for index, media_id in enumerate(ineligible_ids, start=1)
    )
    return pl.DataFrame(
        rows,
        schema=reference_support_manifest_schema(),
        orient="row",
    ).sort(
        [
            "accepted_taxon_key",
            "geo_cluster_id",
            "route",
            "support_split",
            "reference_media_id",
        ],
    )


def _support_row(
    media_id: str,
    *,
    source_path: Path,
    support_split: str | None,
    eligible: bool,
) -> dict[str, object]:
    from biominer.references.admission import strict_reference_admission_policy

    admission_policy = strict_reference_admission_policy()
    assigned_at = NOW - timedelta(days=1)
    assignment = (
        make_reference_split_assignment_fingerprint(
            reference_media_id=media_id,
            split_version="split-v1",
            support_split=str(support_split),
            included=True,
            exclusion_reason=None,
            assigned_by="fixture",
            assigned_at=assigned_at,
        )
        if eligible
        else None
    )
    row: dict[str, object] = {
        "schema_version": REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION,
        "reference_bank_version": "reference-bank-v1",
        "registry_version": "butterflies-v1",
        "reference_media_id": media_id,
        "canonical_reference_media_id": media_id,
        "reference_observation_id": f"observation:{media_id}",
        "reference_admission_mode": admission_policy.mode,
        "reference_admission_policy_version": admission_policy.policy_version,
        "reference_admission_policy_fingerprint": admission_policy.fingerprint,
        "identity_evidence_basis": "human_verified" if eligible else "none",
        "provider_asserted_identity": True,
        "provider_asserted_taxon_key": "1938069",
        "provider_asserted_scientific_name": "Papilio demoleus",
        "provider_dataset_key": "dataset:fixture",
        "provider_quality_status": "research_grade",
        "human_review_status": "completed" if eligible else "rejected",
        "human_verified_identity": eligible,
        "provisional_support": False,
        "statistical_audit_required": False,
        "admission_status": "admitted" if eligible else "excluded",
        "admission_reasons": [
            "strict_human_review_verified" if eligible else "manual_exclusion"
        ],
        "reference_quality_flags": [],
        "route_evidence_basis": "human_verified_review" if eligible else "none",
        "geographic_prototype_eligible": eligible,
        "review_request_id": f"review:{media_id}",
        "review_decision_ids": [f"decision:{media_id}"],
        "reviewer_ids": ["reviewer:fixture"],
        "source": "inaturalist",
        "source_observation_id": f"source:{media_id}",
        "provider_media_id": media_id,
        "source_record_url": f"https://example.test/observations/{media_id}",
        "source_snapshot_version": "fixture-v1",
        "source_dataset_key": "dataset:fixture",
        "accepted_taxon_key": "gbif:1938069",
        "scientific_name": "Papilio demoleus",
        "target_candidate": True,
        "geo_cluster_id": "cluster-a",
        "observer_id": f"observer:{media_id}",
        "observed_at": NOW - timedelta(days=2),
        "latitude": -33.86,
        "longitude": 151.21,
        "source_object_uri": source_path.resolve().as_uri(),
        "image_sha256": _file_sha256(source_path),
        "perceptual_hash": "0123456789abcdef",
        "object_fingerprint": _sha(f"object:{media_id}"),
        "duplicate_group_id": f"duplicate:{media_id}",
        "duplicate_type": "exact",
        "creator": "Fixture Creator",
        "rights_holder": "Fixture Creator",
        "licence": "CC BY 4.0",
        "licence_uri": "https://creativecommons.org/licenses/by/4.0/",
        "licence_policy_status": "accepted",
        "attribution": "Fixture Creator / CC BY 4.0",
        "review_status": "completed" if eligible else "excluded",
        "verification_status": "verified" if eligible else "rejected",
        "target_identity_verified": eligible,
        "life_stage": "adult",
        "visual_domain": "live_field",
        "view": "dorsal",
        "route": "adult_field",
        "support_split": support_split,
        "support_eligible": eligible,
        "exclusion_reasons": [] if eligible else ["manual_exclusion"],
        "split_assignment_fingerprint": assignment,
        "reference_bank_fingerprint": _sha("bank"),
    }
    row["support_row_fingerprint"] = readiness_module._support_row_fingerprint(  # noqa: SLF001 - fixture mirrors persisted contract.
        row
    )
    return row


def _visual_inputs(
    tmp_path: Path,
    manifest: pl.DataFrame,
) -> list[ReferenceVisualInput]:
    result: list[ReferenceVisualInput] = []
    for row in manifest.iter_rows(named=True):
        media_id = str(row["reference_media_id"])
        parsed = urlsplit(str(row["source_object_uri"]))
        source_path = Path(unquote(parsed.path))
        path = tmp_path / f"{media_id}.png"
        path.write_bytes(source_path.read_bytes())
        variant = raw_full_frame_visual_input(_decoded_image(path))
        result.append(
            ReferenceVisualInput.from_variant(
                reference_media_id=media_id,
                source_image_path=source_path,
                image_path=path,
                variant=variant,
            )
        )
    return result


def _image(path: Path, colour: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), colour).save(path, format="PNG")
    return path


def _decoded_image(path: Path) -> DecodedImage:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.load()
        return DecodedImage(
            width=rgb.width,
            height=rgb.height,
            mode="RGB",
            data=rgb.tobytes(),
            source_uri=str(path),
        )


def _vector(index: int) -> list[float]:
    return (
        [1.0, 0.0, 0.0]
        if index % 3 == 0
        else [0.0, 1.0, 0.0]
        if index % 3 == 1
        else [0.0, 0.0, 1.0]
    )


def _permit(manifest: pl.DataFrame) -> ReferenceBankReadinessPermit:
    canonical = manifest.sort(
        [
            "accepted_taxon_key",
            "geo_cluster_id",
            "route",
            "support_split",
            "reference_media_id",
        ]
    )
    identity = ReferenceModelInputIdentity(
        model_name=FakeScorer.model_id,
        model_version="bioclip-2.5-vith14",
        model_revision=REVISION,
        checkpoint_uri="s3://models/bioclip-2.5-vith14/model.safetensors",
        checkpoint_sha256=WEIGHTS_SHA256,
        open_clip_version=FakeScorer.open_clip_version,
        open_clip_config_sha256=OPEN_CLIP_CONFIG_SHA256,
        preprocessing_version=TargetPreprocessingContract().version,
        preprocessing_contract_fingerprint=(TargetPreprocessingContract().fingerprint),
        preprocessing_attestation_fingerprint=(
            _preprocessing_attestation_fingerprint(PREPROCESSING_CONFIG)
        ),
        input_contract_version=FULL_FRAME_VISUAL_INPUT_VERSION,
    )
    return ReferenceBankReadinessPermit(
        status="ready",
        registry_version="butterflies-v1",
        reference_bank_version="reference-bank-v1",
        target_accepted_taxon_key="gbif:1938069",
        policy_fingerprint=_sha("policy"),
        bank_fingerprint=_sha("bank"),
        support_manifest_fingerprint=reference_support_manifest_fingerprint(canonical),
        summary_fingerprint=_sha("summary"),
        split_assignments_fingerprint=_sha("splits"),
        model_input_schema_version=identity.schema_version,
        model_name=identity.model_name,
        model_version=identity.model_version,
        model_revision=identity.model_revision,
        checkpoint_uri=identity.checkpoint_uri,
        checkpoint_sha256=identity.checkpoint_sha256,
        open_clip_version=identity.open_clip_version,
        open_clip_config_sha256=identity.open_clip_config_sha256,
        preprocessing_version=identity.preprocessing_version,
        preprocessing_contract_fingerprint=(
            identity.preprocessing_contract_fingerprint
        ),
        preprocessing_attestation_fingerprint=(
            identity.preprocessing_attestation_fingerprint
        ),
        input_contract_version=identity.input_contract_version,
        model_input_fingerprint=identity.fingerprint,
        readiness_sha256=_sha("readiness"),
        support_manifest_sha256=_sha("support-manifest-file"),
        summary_sha256=_sha("summary-file"),
        permits_reference_embedding=True,
        permits_provisional_scoring=True,
        permits_calibrated_scoring=True,
        permits_scientific_release=True,
        reference_admission_mode="human_verified_strict",
        admission_policy_fingerprint=str(
            canonical["reference_admission_policy_fingerprint"].item(0)
        ),
        provisional_support_count=0,
        human_verified_support_count=canonical.filter(
            pl.col("support_eligible") & pl.col("human_verified_identity")
        ).height,
        statistical_audit_required=False,
    )


def _preprocessing_attestation_fingerprint(
    config: dict[str, object],
) -> str:
    return reference_embeddings_module._preprocessing_attestation_fingerprint(  # noqa: SLF001 - test scorer must attest with the production identity algorithm.
        open_clip_config_sha256=OPEN_CLIP_CONFIG_SHA256,
        open_clip_version=FakeScorer.open_clip_version,
        preprocessing_config=config,
        preprocessing_version=PREPROCESSING_ATTESTATION_VERSION,
    )


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
