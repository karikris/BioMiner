from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.detection.detector_base import DecodedImage
from biominer.references.embedding_routing import run_reference_embedding_routing
from biominer.vision.full_frame_attention import decoded_image_content_hash, raw_full_frame_visual_input


IMAGE = DecodedImage(2, 2, "RGB", bytes(range(12)))
RECORD = {"reference_media_id": "reference-media:" + "1" * 64,
          "source": "gbif", "source_record_hash": "sha256:" + "2" * 64,
          "image_url": "https://example.test/image.jpg"}


class Router:
    model_id = "google/embeddinggemma-2"
    model_version = "fixture"
    fingerprint = "sha256:" + "3" * 64
    prompt_set_fingerprint = "sha256:" + "4" * 64
    config = SimpleNamespace(revision="f" * 40)

    def route_batch(self, images):
        return [{"image_content_hash": decoded_image_content_hash(image),
                 "embedding": [1.0] + [0.0] * 767,
                 "proposed_domain": "adult_butterfly_field", "bioclip_route": "adult_field",
                 "routing_action": "review", "routing_reason": "provisional_embedding_route",
                 "cosine_similarity": 0.7, "similarity_margin": 0.2,
                 "prompt_similarities_json": "[]"} for image in images]


def test_reference_embedding_routes_leave_measurements_unavailable(tmp_path):
    result = run_reference_embedding_routing(records=[RECORD], router=Router(),
        image_loader=lambda _: IMAGE, output_path=tmp_path / "reference_routes.parquet")
    row = pl.read_parquet(result.output_path).row(0, named=True)
    assert row["route_evidence_basis"] == "embeddinggemma2"
    assert row["subject_present"] is None
    assert row["subject_area_ratio"] is None
    assert row["detection_count"] is None
    assert row["reference_admission"] == "review_required"
    assert row["species_identity_decision"] == "not_assessed"
    assert row["raw_visual_input_id"] == raw_full_frame_visual_input(IMAGE).visual_input_id
    fingerprint = row.pop("route_evidence_fingerprint")
    assert fingerprint == canonical_semantic_fingerprint(row)


def test_reference_decode_failure_is_not_an_absent_subject(tmp_path):
    def fail(_):
        raise OSError("fixture decode failure")

    result = run_reference_embedding_routing(records=[RECORD], router=Router(),
        image_loader=fail, output_path=tmp_path / "reference_routes.parquet")
    row = pl.read_parquet(result.output_path).row(0, named=True)
    assert row["subject_present"] is None
    assert row["proposed_domain"] is None
    assert row["inference_status"] == "image_unavailable"
    assert result.image_failures == 1


@pytest.mark.parametrize("update", [{"source": "flickr"}, {"source_record_hash": "missing"}])
def test_reference_routing_requires_source_bound_gbif_evidence(tmp_path, update):
    with pytest.raises(ValueError):
        run_reference_embedding_routing(records=[{**RECORD, **update}], router=Router(),
            image_loader=lambda _: IMAGE, output_path=tmp_path / "reference_routes.parquet")
