"""Tests for durable, one-time Flickr full-frame embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import polars as pl
import pytest

from biominer.detection.detector_base import DecodedImage
from biominer.vision.flickr_embeddings import (
    FLICKR_EMBEDDING_BINDINGS_FILE,
    FLICKR_FULL_FRAME_EMBEDDINGS_FILE,
    FlickrEmbeddingArtifacts,
    load_flickr_embedding_artifacts,
    persist_reusable_flickr_embeddings,
    validate_flickr_embedding_artifacts,
)
from biominer.vision.full_frame_attention import (
    TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
    TARGET_FULL_FRAME_PREPROCESSING,
)
from biominer.vision.target_full_frame import build_target_full_frame_plan


_MODEL_FINGERPRINT = "sha256:" + "b" * 64
_MODEL_ID = "hf-hub:imageomics/bioclip-2"
_MODEL_REVISION = "bioclip-2"
_PREPROCESSING_FINGERPRINT = "sha256:" + "c" * 64
_ROUTING_POLICY_FINGERPRINT = "sha256:" + "a" * 64


class CountingEncoder:
    model_id = _MODEL_ID
    model_revision = _MODEL_REVISION
    image_resize_mode = TARGET_FULL_FRAME_IMAGE_RESIZE_MODE
    preprocessing_contract_fingerprint = TARGET_FULL_FRAME_PREPROCESSING.fingerprint
    preprocessing_fingerprint = _PREPROCESSING_FINGERPRINT

    def __init__(self, *, loads_per_call: int = 1) -> None:
        self.loads_per_call = loads_per_call
        self.model_load_count = 0
        self.batches: list[tuple[DecodedImage, ...]] = []

    def encode_images(self, images: Sequence[DecodedImage]) -> list[list[float]]:
        batch = tuple(images)
        self.batches.append(batch)
        if self.model_load_count == 0:
            self.model_load_count += self.loads_per_call
        return [
            [float(image.data[0] + 1), float(image.width), float(image.height)]
            for image in batch
        ]


def test_persists_one_vector_and_reuses_it_for_photos_routes_and_reruns(
    tmp_path: Path,
) -> None:
    image = _image(1)
    plan = build_target_full_frame_plan(
        detection_rows=[
            _detection_row("photo-1", "adult-1"),
            _detection_row(
                "photo-1",
                "larva-1",
                route="larval",
                detection_route="caterpillar_field",
            ),
            _detection_row("photo-2", "adult-2"),
        ],
        image_loader=lambda _row: image,
    )
    encoder = CountingEncoder()

    first = persist_reusable_flickr_embeddings(
        plan,
        encoder=encoder,
        model_id=_MODEL_ID,
        model_revision=_MODEL_REVISION,
        model_fingerprint=_MODEL_FINGERPRINT,
        preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
        output_dir=tmp_path / "cache",
    )

    assert len(plan.visual_inputs) == 1
    assert first.artifacts.embeddings.height == 1
    assert first.artifacts.photo_bindings.height == 2
    assert len(first.embedded_plan.scoring_unit_references) == 3
    assert (
        len({item.embedding_id for item in first.embedded_plan.scoring_unit_references})
        == 1
    )
    assert first.cache_hits == 0
    assert first.cache_misses == 1
    assert first.encoder_calls == 1
    assert first.images_encoded == 1
    assert first.encoder_model_load_count_delta == 1
    assert encoder.model_load_count == 1
    assert len(encoder.batches) == 1
    assert first.embeddings_path.name == FLICKR_FULL_FRAME_EMBEDDINGS_FILE
    assert first.photo_bindings_path.name == FLICKR_EMBEDDING_BINDINGS_FILE

    second = persist_reusable_flickr_embeddings(
        plan,
        encoder=encoder,
        model_id=_MODEL_ID,
        model_revision=_MODEL_REVISION,
        model_fingerprint=_MODEL_FINGERPRINT,
        preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
        output_dir=tmp_path / "cache",
    )

    assert second.cache_hits == 1
    assert second.cache_misses == 0
    assert second.encoder_calls == 0
    assert second.images_encoded == 0
    assert second.encoder_model_load_count_delta == 0
    assert len(encoder.batches) == 1
    assert second.artifacts.embeddings.equals(first.artifacts.embeddings)
    assert second.artifacts.photo_bindings.equals(first.artifacts.photo_bindings)
    loaded = load_flickr_embedding_artifacts(tmp_path / "cache")
    assert loaded.embeddings.equals(first.artifacts.embeddings)
    assert loaded.photo_bindings.equals(first.artifacts.photo_bindings)

    drifted_encoder = CountingEncoder()
    drifted_encoder.model_id = "hf-hub:imageomics/another-model"
    with pytest.raises(ValueError, match="bound to another model identity"):
        persist_reusable_flickr_embeddings(
            plan,
            encoder=drifted_encoder,
            model_id=drifted_encoder.model_id,
            model_revision=_MODEL_REVISION,
            model_fingerprint=_MODEL_FINGERPRINT,
            preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
            output_dir=tmp_path / "cache",
        )
    assert drifted_encoder.batches == []


def test_changed_content_encodes_only_the_new_visual_identity(tmp_path: Path) -> None:
    images = {"photo-1": _image(1), "photo-2": _image(2)}
    encoder = CountingEncoder()
    first_plan = _plan(images)
    first = persist_reusable_flickr_embeddings(
        first_plan,
        encoder=encoder,
        model_id=_MODEL_ID,
        model_revision=_MODEL_REVISION,
        model_fingerprint=_MODEL_FINGERPRINT,
        preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
        output_dir=tmp_path / "cache",
    )
    assert first.images_encoded == 2

    images["photo-2"] = _image(3)
    changed_plan = _plan(images)
    changed = persist_reusable_flickr_embeddings(
        changed_plan,
        encoder=encoder,
        model_id=_MODEL_ID,
        model_revision=_MODEL_REVISION,
        model_fingerprint=_MODEL_FINGERPRINT,
        preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
        output_dir=tmp_path / "cache",
    )

    assert changed.cache_hits == 1
    assert changed.cache_misses == 1
    assert changed.images_encoded == 1
    assert changed.encoder_calls == 1
    assert changed.encoder_model_load_count_delta == 0
    assert changed.artifacts.embeddings.height == 3
    assert changed.artifacts.photo_bindings.height == 3
    assert [len(batch) for batch in encoder.batches] == [2, 1]


def test_loader_rejects_tampered_vectors_and_binding_foreign_keys(
    tmp_path: Path,
) -> None:
    plan = _plan({"photo-1": _image(1)})
    result = persist_reusable_flickr_embeddings(
        plan,
        encoder=CountingEncoder(),
        model_id=_MODEL_ID,
        model_revision=_MODEL_REVISION,
        model_fingerprint=_MODEL_FINGERPRINT,
        preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
        output_dir=tmp_path / "cache",
    )
    tampered_embeddings = result.artifacts.embeddings.with_columns(
        pl.Series(
            "embedding",
            [[9.0, 9.0, 9.0]],
            dtype=pl.List(pl.Float64),
        )
    )
    tampered_embeddings.write_parquet(result.embeddings_path)

    with pytest.raises(ValueError, match="norm mismatch"):
        load_flickr_embedding_artifacts(tmp_path / "cache")

    invalid_bindings = result.artifacts.photo_bindings.with_columns(
        pl.lit("sha256:" + "f" * 64).alias("embedding_id")
    )
    with pytest.raises(ValueError):
        validate_flickr_embedding_artifacts(
            FlickrEmbeddingArtifacts(
                embeddings=result.artifacts.embeddings,
                photo_bindings=invalid_bindings,
            )
        )


def test_one_embedding_batch_cannot_load_encoder_model_twice(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loaded the model more than once"):
        persist_reusable_flickr_embeddings(
            _plan({"photo-1": _image(1)}),
            encoder=CountingEncoder(loads_per_call=2),
            model_id=_MODEL_ID,
            model_revision=_MODEL_REVISION,
            model_fingerprint=_MODEL_FINGERPRINT,
            preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
            output_dir=tmp_path / "cache",
        )

    assert not (tmp_path / "cache" / FLICKR_FULL_FRAME_EMBEDDINGS_FILE).exists()
    assert not (tmp_path / "cache" / FLICKR_EMBEDDING_BINDINGS_FILE).exists()


def _plan(images: dict[str, DecodedImage]):
    return build_target_full_frame_plan(
        detection_rows=[
            _detection_row(photo_id, f"det-{photo_id}") for photo_id in images
        ],
        image_loader=lambda row: images[str(row["flickr_photo_id"])],
    )


def _image(value: int) -> DecodedImage:
    return DecodedImage(
        width=2,
        height=2,
        mode="RGB",
        data=bytes([value] * 12),
        source_uri=f"memory://image-{value}",
    )


def _detection_row(
    photo_id: str,
    detection_id: str,
    *,
    route: str = "adult_field",
    detection_route: str = "adult_butterfly_field",
) -> dict[str, Any]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": "sha256:" + "f" * 64,
        "detection_id": detection_id,
        "detection_status": "detected",
        "detector_score": 0.9,
        "detector_label": "butterfly_like",
        "bbox_xyxy": [0.0, 0.0, 2.0, 2.0],
        "bbox_xyxyn": [0.0, 0.0, 1.0, 1.0],
        "mask_polygon_xyn": None,
        "detection_route": detection_route,
        "routing_action": "score",
        "bioclip_route": route,
        "routing_policy_version": "detection-routing-policy-v1",
        "routing_policy_fingerprint": _ROUTING_POLICY_FINGERPRINT,
        "schema_version": "object-detection-v2",
    }
