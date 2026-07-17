from __future__ import annotations

from pathlib import Path

import polars as pl

from biominer.detection.detector_base import DecodedImage
from biominer.vision.score_inputs import BIOCLIP_SCORE_INPUT_SCHEMA, materialize_bioclip_score_inputs
from factories import canonical_records, flickr_source_record, object_detection_row


_ROUTING_POLICY_FINGERPRINT = "sha256:" + "b" * 64


def test_materialize_routed_score_inputs_propagates_route_and_policy_identity(
    tmp_path: Path,
) -> None:
    records = canonical_records(
        flickr_source_record("photo-adult"),
        flickr_source_record("photo-review"),
    )
    detections = pl.DataFrame(
        [
            {
                **object_detection_row(
                    "photo-adult",
                    detection_id="det-adult",
                    crop_hash="sha256:adult",
                ),
                **_routing_fields(
                    detection_route="adult_butterfly_field",
                    routing_action="score",
                    bioclip_route="adult_field",
                    routing_priority="standard",
                ),
            },
            {
                **object_detection_row(
                    "photo-review",
                    detection_id="det-review",
                    crop_hash="sha256:review",
                ),
                **_routing_fields(
                    detection_route="ambiguous_visual_domain",
                    routing_action="review",
                    bioclip_route="larval",
                    routing_priority="low",
                ),
            },
        ]
    )
    loaded_for: list[str] = []

    def image_loader(item: dict[str, object]) -> DecodedImage:
        loaded_for.append(str(item["detection_id"]))
        return _decoded_image()

    result = materialize_bioclip_score_inputs(
        canonical_records=records,
        detections=detections,
        image_loader=image_loader,
        temp_dir=tmp_path,
        crop_target_px=3,
    )

    assert result.frame.schema == BIOCLIP_SCORE_INPUT_SCHEMA
    assert result.frame["detection_id"].to_list() == ["det-adult"]
    assert result.frame.select(
        "detection_route",
        "routing_action",
        "bioclip_route",
        "routing_priority",
        "routing_reason",
        "routing_policy_version",
        "routing_policy_fingerprint",
    ).row(0, named=True) == _routing_fields(
        detection_route="adult_butterfly_field",
        routing_action="score",
        bioclip_route="adult_field",
        routing_priority="standard",
    )
    assert {
        name: result.items[0][name]
        for name in (
            "detection_route",
            "routing_action",
            "bioclip_route",
            "routing_priority",
            "routing_reason",
            "routing_policy_version",
            "routing_policy_fingerprint",
        )
    } == _routing_fields(
        detection_route="adult_butterfly_field",
        routing_action="score",
        bioclip_route="adult_field",
        routing_priority="standard",
    )
    assert loaded_for == ["det-adult"]

    result.cleanup()


def _routing_fields(
    *,
    detection_route: str,
    routing_action: str,
    bioclip_route: str,
    routing_priority: str,
) -> dict[str, object]:
    return {
        "detection_route": detection_route,
        "routing_action": routing_action,
        "bioclip_route": bioclip_route,
        "routing_priority": routing_priority,
        "routing_reason": "test routing decision",
        "routing_policy_version": "detection-routing-v1",
        "routing_policy_fingerprint": _ROUTING_POLICY_FINGERPRINT,
    }


def _decoded_image() -> DecodedImage:
    pixels = bytes(
        value
        for y in range(4)
        for x in range(4)
        for value in ((x * 40) % 256, (y * 40) % 256, ((x + y) * 20) % 256)
    )
    return DecodedImage(width=4, height=4, mode="RGB", data=pixels, source_uri="memory://score-input")
