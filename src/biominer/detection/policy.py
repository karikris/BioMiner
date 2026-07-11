from __future__ import annotations

from dataclasses import dataclass, field, fields, replace


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
    adaptive_batching: bool = False
    min_detector_batch_size: int = 1


@dataclass(frozen=True)
class VisionRuntimeSettings:
    profile_name: str = "default"
    device: str = "auto"
    yolo_checkpoint: str = "yoloe-26s-seg.pt"
    yolo_imgsz: int = 640
    yolo_conf: float = 0.20
    yolo_iou: float = 0.50
    yolo_max_det: int = 8
    detector_batch_size: int = 4
    crop_batch_size: int = 24
    bioclip_preprocess_workers: int = 1
    crop_padding_ratio: float = 0.12
    crop_target_px: int = 336
    bioclip_model: str = "imageomics/bioclip-2.5-vith14"
    bioclip_top_k: int = 10
    parquet_compression: str = "zstd"
    parquet_part_rows: int = 10000
    delete_images_after_commit: bool = True
    retain_debug_crops: bool = False
    debug_crop_limit: int = 500
    image_max_side_px: int = 1280
    adaptive_batching: bool = False
    yolo_sidecar_transport: str = "json_b64"
    min_detector_batch_size: int = 1
    min_crop_batch_size: int = 1
    max_detector_batch_size: int = 16
    max_crop_batch_size: int = 24
    mps_memory_safety_margin_mb: int | None = None

    def with_overrides(self, **overrides: object) -> VisionRuntimeSettings:
        allowed = {field.name for field in fields(VisionRuntimeSettings)}
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise ValueError("unknown vision runtime setting override(s): " + ", ".join(unknown))
        cleaned = {key: value for key, value in overrides.items() if value is not None}
        return validate_vision_runtime_settings(replace(self, **cleaned))

    def to_detection_policy(self, base: DetectionPolicy | None = None) -> DetectionPolicy:
        active = base or DetectionPolicy()
        return DetectionPolicy(
            backend=active.backend,
            box_score_threshold=self.yolo_conf,
            nms_iou_threshold=self.yolo_iou,
            min_box_area_ratio=active.min_box_area_ratio,
            max_boxes_per_image=self.yolo_max_det,
            bioclip_eligible_labels=active.bioclip_eligible_labels,
            crop_padding_ratio=self.crop_padding_ratio,
            image_max_side_px=self.image_max_side_px,
            crop_target_px=self.crop_target_px,
            retain_debug_crops=self.retain_debug_crops,
            debug_crop_limit=self.debug_crop_limit,
        )

    def to_detection_run_policy(self, base: DetectionRunPolicy | None = None) -> DetectionRunPolicy:
        active = base or DetectionRunPolicy()
        return DetectionRunPolicy(
            download_workers=active.download_workers,
            decode_workers=active.decode_workers,
            detector_workers=active.detector_workers,
            max_inflight_images=active.max_inflight_images,
            max_inflight_crops=active.max_inflight_crops,
            detector_batch_size=self.detector_batch_size,
            crop_batch_size=self.crop_batch_size,
            parquet_batch_rows=self.parquet_part_rows,
            adaptive_batching=self.adaptive_batching,
            min_detector_batch_size=self.min_detector_batch_size,
        )


@dataclass(frozen=True)
class RuntimeProfile:
    profile_name: str
    detection_policy: DetectionPolicy
    run_policy: DetectionRunPolicy
    vision_settings: VisionRuntimeSettings = field(default_factory=VisionRuntimeSettings)
    bioclip_workers: int = 1
    text_embedding_batch_size: int = 256
    worker_shard_target_mb: int = 64
    compacted_shard_target_mb: int = 256


MAC_M5PRO_64GB_SETTINGS = VisionRuntimeSettings(
    profile_name="mac_m5pro_64gb",
    device="mps",
    yolo_checkpoint="yoloe-26s-seg.pt",
    yolo_imgsz=768,
    yolo_conf=0.20,
    yolo_iou=0.50,
    yolo_max_det=8,
    detector_batch_size=16,
    crop_batch_size=24,
    bioclip_preprocess_workers=4,
    crop_padding_ratio=0.08,
    crop_target_px=336,
    bioclip_model="hf-hub:imageomics/bioclip-2.5-vith14",
    bioclip_top_k=10,
    parquet_compression="zstd",
    parquet_part_rows=500,
    delete_images_after_commit=True,
    retain_debug_crops=False,
)

MAC_M5PRO_64GB_PROFILE = RuntimeProfile(
    profile_name=MAC_M5PRO_64GB_SETTINGS.profile_name,
    detection_policy=MAC_M5PRO_64GB_SETTINGS.to_detection_policy(),
    run_policy=MAC_M5PRO_64GB_SETTINGS.to_detection_run_policy(
        DetectionRunPolicy(
            download_workers=4,
            decode_workers=4,
            detector_workers=1,
            max_inflight_images=32,
            max_inflight_crops=96,
        )
    ),
    vision_settings=MAC_M5PRO_64GB_SETTINGS,
    bioclip_workers=1,
    text_embedding_batch_size=256,
    worker_shard_target_mb=64,
    compacted_shard_target_mb=256,
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


def vision_runtime_settings(profile_name: str) -> VisionRuntimeSettings:
    return runtime_profile(profile_name).vision_settings


def validate_vision_runtime_settings(settings: VisionRuntimeSettings) -> VisionRuntimeSettings:
    _positive_int(settings.yolo_imgsz, "yolo_imgsz")
    _positive_int(settings.yolo_max_det, "yolo_max_det")
    _positive_int(settings.detector_batch_size, "detector_batch_size")
    _positive_int(settings.bioclip_preprocess_workers, "bioclip_preprocess_workers")
    _positive_int(settings.crop_batch_size, "crop_batch_size")
    _positive_int(settings.min_detector_batch_size, "min_detector_batch_size")
    _positive_int(settings.min_crop_batch_size, "min_crop_batch_size")
    _positive_int(settings.max_detector_batch_size, "max_detector_batch_size")
    _positive_int(settings.max_crop_batch_size, "max_crop_batch_size")
    _positive_int(settings.crop_target_px, "crop_target_px")
    _positive_int(settings.bioclip_top_k, "bioclip_top_k")
    _positive_int(settings.parquet_part_rows, "parquet_part_rows")
    _positive_int(settings.image_max_side_px, "image_max_side_px")
    _non_negative_int(settings.debug_crop_limit, "debug_crop_limit")
    if settings.mps_memory_safety_margin_mb is not None:
        _positive_int(settings.mps_memory_safety_margin_mb, "mps_memory_safety_margin_mb")
    if settings.min_detector_batch_size > settings.max_detector_batch_size:
        raise ValueError("min_detector_batch_size must be <= max_detector_batch_size")
    if settings.min_crop_batch_size > settings.max_crop_batch_size:
        raise ValueError("min_crop_batch_size must be <= max_crop_batch_size")
    if not settings.min_detector_batch_size <= settings.detector_batch_size <= settings.max_detector_batch_size:
        raise ValueError("detector_batch_size must be between min_detector_batch_size and max_detector_batch_size")
    if not settings.min_crop_batch_size <= settings.crop_batch_size <= settings.max_crop_batch_size:
        raise ValueError("crop_batch_size must be between min_crop_batch_size and max_crop_batch_size")
    if not 0.0 <= float(settings.crop_padding_ratio) <= 0.5:
        raise ValueError("crop_padding_ratio must be between 0.0 and 0.5")
    if not 0.0 <= float(settings.yolo_conf) <= 1.0:
        raise ValueError("yolo_conf must be between 0.0 and 1.0")
    if not 0.0 <= float(settings.yolo_iou) <= 1.0:
        raise ValueError("yolo_iou must be between 0.0 and 1.0")
    if settings.yolo_sidecar_transport not in {"json_b64", "image_path"}:
        raise ValueError("yolo_sidecar_transport must be one of: json_b64, image_path")
    return settings


def _positive_int(value: object, name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _non_negative_int(value: object, name: str) -> None:
    if int(value) < 0:
        raise ValueError(f"{name} must be non-negative")


def detection_is_bioclip_eligible(row: dict[str, object], policy: DetectionPolicy | None = None) -> bool:
    active_policy = policy or DetectionPolicy()
    if str(row.get("detection_status") or "") != "detected":
        return False
    label = str(row.get("detector_label") or "")
    return label in set(active_policy.bioclip_eligible_labels)
