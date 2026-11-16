from __future__ import annotations

import pytest

from biominer.detection.detector_base import DecodedImage
from biominer.vision.full_frame_attention import (
    decoded_image_content_hash, TARGET_FULL_FRAME_IMAGE_RESIZE_MODE, TARGET_FULL_FRAME_PREPROCESSING,
)
from biominer.vision.target_full_frame import (
    build_embedding_routed_full_frame_plan, encode_target_full_frame_plan,
    generate_target_full_frame_attention_variants,
)


def image():
    return DecodedImage(2, 2, "RGB", bytes(range(12)))


def row():
    return {"schema_version": "flickr-embedding-screening-v1", "source": "flickr",
            "flickr_photo_id": "photo-1", "source_record_hash": "sha256:" + "1" * 64,
            "image_content_hash": decoded_image_content_hash(image()),
            "model_fingerprint": "sha256:" + "2" * 64,
            "prompt_set_fingerprint": "sha256:" + "3" * 64,
            "evidence_kind": "full_frame_embedding", "routing_action": "review",
            "bioclip_route": "adult_field", "proposed_domain": "adult_butterfly_field",
            "cosine_similarity": 0.7, "similarity_margin": 0.2,
            "subject_present": None, "subject_area_ratio": None,
            "detection_count": None, "bbox_xyxy": None, "mask_polygon_xyn": None}


def test_embedding_route_preserves_raw_image_and_has_no_detection_geometry():
    plan = build_embedding_routed_full_frame_plan(embedding_rows=[row()], image_loader=lambda _: image())
    assert plan.visual_inputs[0].image == image()
    unit = plan.scoring_units[0]
    assert unit.route == "adult_field"
    assert unit.detections == ()
    assert unit.detector_boxes_xyxy == ()
    assert unit.embedding_route_evidence[0].cosine_similarity == 0.7


def test_embedding_route_can_encode_bioclip_full_frame_independently():
    class Encoder:
        image_resize_mode = TARGET_FULL_FRAME_IMAGE_RESIZE_MODE
        preprocessing_contract_fingerprint = TARGET_FULL_FRAME_PREPROCESSING.fingerprint
        preprocessing_fingerprint = "sha256:" + "4" * 64
        calls = 0

        def encode_images(self, images):
            self.calls += 1
            return [[1.0, 2.0, 3.0] for _ in images]

    plan = build_embedding_routed_full_frame_plan(embedding_rows=[row()], image_loader=lambda _: image())
    encoder = Encoder()
    result = encode_target_full_frame_plan(plan, encoder=encoder,
        model_fingerprint="sha256:" + "5" * 64,
        preprocessing_fingerprint=encoder.preprocessing_fingerprint)
    assert encoder.calls == 1
    assert len(result.embeddings) == 1


def test_embedding_route_rejects_stale_image_content():
    with pytest.raises(ValueError, match="content|image"):
        build_embedding_routed_full_frame_plan(embedding_rows=[row()], image_loader=lambda _: DecodedImage(1, 1, "RGB", b"\0\0\0"))


def test_embedding_routes_cannot_generate_spatial_attention():
    plan = build_embedding_routed_full_frame_plan(embedding_rows=[row()], image_loader=lambda _: image())
    with pytest.raises(ValueError, match="localization|spatial"):
        generate_target_full_frame_attention_variants(plan, scoring_unit_id=plan.scoring_units[0].scoring_unit_id)


def test_embedding_route_identity_changes_with_routing_model():
    first = build_embedding_routed_full_frame_plan(embedding_rows=[row()], image_loader=lambda _: image())
    changed = {**row(), "model_fingerprint": "sha256:" + "6" * 64}
    second = build_embedding_routed_full_frame_plan(embedding_rows=[changed], image_loader=lambda _: image())
    assert first.scoring_units[0].scoring_unit_id != second.scoring_units[0].scoring_unit_id
    assert first.visual_inputs[0].visual_input_id == second.visual_inputs[0].visual_input_id


@pytest.mark.parametrize("domain", ["pinned_specimen", "no_relevant_organism", "caterpillar_field"])
def test_embedding_plan_rejects_incompatible_adult_route(domain):
    with pytest.raises(ValueError, match="domain|route"):
        build_embedding_routed_full_frame_plan(embedding_rows=[{**row(), "proposed_domain": domain}], image_loader=lambda _: image())


def test_embedding_plan_builds_flickr_scoring_artifacts_without_organism_measurements():
    from biominer.flickr_fetch.scoring_units import build_flickr_scoring_unit_artifacts, validate_flickr_scoring_unit_artifacts

    plan = build_embedding_routed_full_frame_plan(embedding_rows=[row()], image_loader=lambda _: image())
    artifacts = build_flickr_scoring_unit_artifacts(plan, run_id="embedding-run", associations=[{
        "source": "flickr", "flickr_photo_id": "photo-1", "association_kind": "query",
        "association_source": "flickr", "association_source_id": "query-1",
        "flickr_query_id": "query-1",
    }])
    validate_flickr_scoring_unit_artifacts(artifacts)
    scoring = artifacts.scoring_units.row(0, named=True)
    assert scoring["scoring_unit_kind"] == "full_frame_embedding_route"
    assert scoring["detection_count"] is None
    assert scoring["detection_ids"] is None
    assert scoring["routing_model_fingerprint"] == row()["model_fingerprint"]
    assert artifacts.associations.height == 1
