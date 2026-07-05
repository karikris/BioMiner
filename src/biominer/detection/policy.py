from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionPolicy:
    backend: str = "yoloe26"
    box_score_threshold: float = 0.20
    nms_iou_threshold: float = 0.50
    min_box_area_ratio: float = 0.0005
    max_boxes_per_image: int = 8
    bioclip_eligible_labels: tuple[str, ...] = ("butterfly_like",)
    crop_padding_ratio: float = 0.12
    image_max_side_px: int = 1280
    crop_target_px: int = 336
    retain_debug_crops: bool = False
    debug_crop_limit: int = 500


@dataclass(frozen=True)
class DetectionRunPolicy:
    download_workers: int = 4
    decode_workers: int = 4
    detector_workers: int = 1
    max_inflight_images: int = 32
    max_inflight_crops: int = 96
    detector_batch_size: int = 4
    crop_batch_size: int = 24
    parquet_batch_rows: int = 10000


@dataclass(frozen=True)
class RuntimeProfile:
    profile_name: str
    detection_policy: DetectionPolicy
    run_policy: DetectionRunPolicy
    bioclip_workers: int = 1
    text_embedding_batch_size: int = 256
    worker_shard_target_mb: int = 64
    compacted_shard_target_mb: int = 256


MAC_M5PRO_64GB_PROFILE = RuntimeProfile(
    profile_name="mac_m5pro_64gb",
    detection_policy=DetectionPolicy(
        image_max_side_px=1280,
        crop_target_px=336,
        retain_debug_crops=False,
    ),
    run_policy=DetectionRunPolicy(
        download_workers=4,
        decode_workers=4,
        detector_workers=1,
        max_inflight_images=32,
        max_inflight_crops=96,
        detector_batch_size=4,
        crop_batch_size=24,
    ),
)

RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    MAC_M5PRO_64GB_PROFILE.profile_name: MAC_M5PRO_64GB_PROFILE,
}


def runtime_profile(profile_name: str) -> RuntimeProfile:
    try:
        return RUNTIME_PROFILES[profile_name]
    except KeyError as exc:
        known = ", ".join(sorted(RUNTIME_PROFILES))
        raise ValueError(f"unknown runtime profile {profile_name!r}; expected one of: {known}") from exc


def detection_is_bioclip_eligible(row: dict[str, object], policy: DetectionPolicy | None = None) -> bool:
    active_policy = policy or DetectionPolicy()
    if str(row.get("detection_status") or "") != "detected":
        return False
    label = str(row.get("detector_label") or "")
    return label in set(active_policy.bioclip_eligible_labels)
