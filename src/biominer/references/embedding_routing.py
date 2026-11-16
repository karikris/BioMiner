"""Source-bound, raw-frame reference route suggestions requiring separate QA."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.detection.detector_base import DecodedImage
from biominer.storage.parquet import write_parquet_batches
from biominer.vision.full_frame_attention import decoded_image_content_hash, raw_full_frame_visual_input


REFERENCE_EMBEDDING_ROUTING_VERSION = "reference-embedding-routing-v1"
REFERENCE_EMBEDDING_ROUTE_SCHEMA = {
    "schema_version": pl.String, "reference_media_id": pl.String,
    "source": pl.String, "source_record_hash": pl.String,
    "route_evidence_basis": pl.String, "model_id": pl.String,
    "model_revision": pl.String, "model_version": pl.String,
    "model_fingerprint": pl.String, "prompt_set_fingerprint": pl.String,
    "inference_status": pl.String, "proposed_domain": pl.String,
    "bioclip_route": pl.String, "routing_action": pl.String, "routing_reason": pl.String,
    "cosine_similarity": pl.Float64, "similarity_margin": pl.Float64,
    "prompt_similarities_json": pl.String, "embedding": pl.List(pl.Float64),
    "raw_visual_input_id": pl.String, "raw_visual_content_hash": pl.String,
    "raw_transformation_fingerprint": pl.String,
    "subject_present": pl.Boolean, "subject_area_ratio": pl.Float64,
    "detection_count": pl.UInt32, "localization_status": pl.String,
    "reference_admission": pl.String, "species_identity_decision": pl.String,
    "failure_reason": pl.String, "routed_at": pl.String,
    "route_evidence_fingerprint": pl.String,
}


@dataclass(frozen=True, slots=True)
class ReferenceEmbeddingRoutingResult:
    output_path: Path
    records_seen: int
    image_failures: int


def run_reference_embedding_routing(
    *,
    records: Iterable[Mapping[str, Any]],
    router: Any,
    image_loader: Callable[[dict[str, Any]], DecodedImage],
    output_path: str | Path,
    batch_size: int = 8,
) -> ReferenceEmbeddingRoutingResult:
    """Persist proposals with null localization/subject QA and no auto-admission.

    The versioned result is deliberately distinct from measured YOLOE reference
    routes. It cannot satisfy the historical automatic admission compiler's
    geometry-dependent schema or quality gates.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    seen: set[str] = set()
    counters = {"records": 0, "image_failures": 0}

    def batches():
        pending = []
        for raw_record in records:
            record = dict(raw_record)
            media = record.get("reference_media_id")
            source_hash = record.get("source_record_hash")
            if not isinstance(media, str) or not re.fullmatch(r"reference-media:[0-9a-f]{64}", media):
                raise ValueError("reference routing requires a canonical reference_media_id")
            if media in seen:
                raise ValueError("duplicate reference_media_id")
            if record.get("source") != "gbif" or not isinstance(source_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", source_hash):
                raise ValueError("reference routing requires source-bound GBIF evidence")
            seen.add(media)
            counters["records"] += 1
            pending.append(record)
            if len(pending) >= batch_size:
                yield _reference_batch(pending, router, image_loader, counters)
                pending = []
        if pending:
            yield _reference_batch(pending, router, image_loader, counters)

    output = write_parquet_batches(batches(), output_path, schema=REFERENCE_EMBEDDING_ROUTE_SCHEMA)
    return ReferenceEmbeddingRoutingResult(output, counters["records"], counters["image_failures"])


def _reference_batch(records, router, loader, counters):
    rows = []
    loaded = []
    for record in records:
        base = {
            "schema_version": REFERENCE_EMBEDDING_ROUTING_VERSION,
            "reference_media_id": record["reference_media_id"], "source": record["source"],
            "source_record_hash": record["source_record_hash"], "route_evidence_basis": "embeddinggemma2",
            "model_id": router.model_id, "model_revision": router.config.revision,
            "model_version": router.model_version, "model_fingerprint": router.fingerprint,
            "prompt_set_fingerprint": router.prompt_set_fingerprint,
            "inference_status": "image_unavailable", "routing_action": "review",
            "routing_reason": "image_unavailable", "localization_status": "unavailable",
            "reference_admission": "review_required", "species_identity_decision": "not_assessed",
            "routed_at": datetime.now(UTC).isoformat(),
        }
        try:
            image = loader(record)
            if not isinstance(image, DecodedImage):
                raise TypeError("image_loader must return a DecodedImage")
        except Exception as exc:
            counters["image_failures"] += 1
            rows.append({**base, "failure_reason": f"{type(exc).__name__}: {exc}"})
            continue
        loaded.append((base, image))
    if loaded:
        evidence = router.route_batch([image for _, image in loaded])
        if len(evidence) != len(loaded):
            raise ValueError("reference embedding batch cardinality mismatch")
        for (base, image), result in zip(loaded, evidence, strict=True):
            if result.get("image_content_hash") != decoded_image_content_hash(image):
                raise ValueError("reference embedding image content mismatch")
            if result.get("routing_action") != "review":
                raise ValueError("reference embedding proposals require review")
            for name in ("subject_present", "subject_area_ratio", "detection_count", "bbox_xyxy", "mask_polygon_xyn", "detector_score"):
                if result.get(name) is not None:
                    raise ValueError(f"reference embeddings cannot measure {name}")
            raw = raw_full_frame_visual_input(image)
            rows.append({**base, "inference_status": "image_routed",
                **{name: result[name] for name in ("proposed_domain", "bioclip_route", "routing_reason",
                    "cosine_similarity", "similarity_margin", "prompt_similarities_json", "embedding")},
                "raw_visual_input_id": raw.visual_input_id,
                "raw_visual_content_hash": raw.visual_content_hash,
                "raw_transformation_fingerprint": raw.transformation_fingerprint,
            })
    normalized = []
    for value in rows:
        row = {key: value.get(key) for key in REFERENCE_EMBEDDING_ROUTE_SCHEMA}
        row.pop("route_evidence_fingerprint")
        row["route_evidence_fingerprint"] = canonical_semantic_fingerprint(row)
        normalized.append(row)
    return pl.DataFrame(normalized, schema=REFERENCE_EMBEDDING_ROUTE_SCHEMA).sort("reference_media_id")
