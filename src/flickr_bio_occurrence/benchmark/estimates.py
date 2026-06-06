from __future__ import annotations

import math
from typing import Any


def estimate_production_from_report(
    report: dict[str, Any],
    *,
    target_records_with_images: int,
    api_call_target: int | None = None,
    soft_api_calls_per_hour: int | None = None,
) -> dict[str, float | int | bool | None]:
    measured_records = int(report.get("actual_unique_records") or 0)
    measured_api_calls = int(report.get("api_calls_made") or report.get("work_items_called") or 0)
    if measured_records <= 0:
        raise ValueError("actual_unique_records must be greater than zero to estimate production throughput")
    if measured_api_calls <= 0:
        raise ValueError("api_calls_made or work_items_called must be greater than zero to estimate production throughput")

    timings = report.get("step_timings_seconds", {})
    api_policy = report.get("api_policy", {})
    effective_soft_cap = int(soft_api_calls_per_hour or api_policy.get("soft_api_calls_per_hour") or 3200)
    effective_api_call_target = int(api_call_target or effective_soft_cap)
    fetch_seconds = float(timings.get("flickr_fetch", 0.0))
    vision_seconds = float(timings.get("vision_classification", 0.0))
    nonvision_record_seconds = _nonvision_record_seconds(timings, measured_records)
    records_per_api_call = measured_records / measured_api_calls
    api_calls_needed = target_records_with_images / records_per_api_call
    records_expected = effective_api_call_target * records_per_api_call
    fetch_seconds_per_call = fetch_seconds / measured_api_calls
    vision_seconds_per_image = vision_seconds / measured_records if vision_seconds else 0.0
    end_to_end_seconds = (
        fetch_seconds_per_call * api_calls_needed
        + nonvision_record_seconds * target_records_with_images
        + vision_seconds_per_image * target_records_with_images
    )

    return {
        "target_records_with_images": target_records_with_images,
        "api_call_target": effective_api_call_target,
        "soft_api_calls_per_hour": effective_soft_cap,
        "hard_api_calls_per_hour": _optional_int(api_policy.get("hard_api_calls_per_hour")),
        "hard_photo_records_per_hour": _optional_int(api_policy.get("hard_photo_records_per_hour")),
        "measured_api_calls": measured_api_calls,
        "measured_records_with_images": measured_records,
        "observed_records_per_api_call": records_per_api_call,
        "fetch_seconds_for_measured_api_calls": fetch_seconds,
        "vision_seconds_per_image": vision_seconds_per_image,
        "estimated_vision_seconds_for_100_images": vision_seconds_per_image * 100,
        "estimated_metadata_seconds_for_api_call_target": fetch_seconds_per_call * effective_api_call_target,
        "estimated_nonvision_pipeline_seconds_for_target_records": nonvision_record_seconds * target_records_with_images,
        "estimated_end_to_end_seconds_for_target_records_with_images": end_to_end_seconds,
        "api_calls_needed_for_target_records_at_observed_yield": api_calls_needed,
        "records_expected_at_api_call_target_observed_yield": records_expected,
        "exceeds_single_soft_cap_at_observed_yield": math.ceil(api_calls_needed) > effective_soft_cap,
    }


def estimate_combined_production(
    metadata_report: dict[str, Any],
    vision_report: dict[str, Any],
    *,
    target_records_with_images: int,
    api_call_target: int | None = None,
    soft_api_calls_per_hour: int | None = None,
) -> dict[str, float | int | bool | None]:
    metadata_records = int(metadata_report.get("actual_unique_records") or 0)
    metadata_api_calls = int(metadata_report.get("api_calls_made") or metadata_report.get("work_items_called") or 0)
    vision_images = int(vision_report.get("actual_unique_records") or 0)
    if metadata_records <= 0:
        raise ValueError("metadata actual_unique_records must be greater than zero")
    if metadata_api_calls <= 0:
        raise ValueError("metadata api_calls_made or work_items_called must be greater than zero")
    if vision_images <= 0:
        raise ValueError("vision actual_unique_records must be greater than zero")

    metadata_timings = metadata_report.get("step_timings_seconds", {})
    vision_timings = vision_report.get("step_timings_seconds", {})
    api_policy = metadata_report.get("api_policy", {})
    effective_soft_cap = int(soft_api_calls_per_hour or api_policy.get("soft_api_calls_per_hour") or 3200)
    effective_api_call_target = int(api_call_target or effective_soft_cap)
    records_per_api_call = metadata_records / metadata_api_calls
    api_calls_needed = target_records_with_images / records_per_api_call
    fetch_seconds_per_call = float(metadata_timings.get("flickr_fetch", 0.0)) / metadata_api_calls
    nonvision_record_seconds = _nonvision_record_seconds(metadata_timings, metadata_records)
    vision_seconds_per_image = float(vision_timings.get("vision_classification", 0.0)) / vision_images
    end_to_end_seconds = (
        fetch_seconds_per_call * api_calls_needed
        + nonvision_record_seconds * target_records_with_images
        + vision_seconds_per_image * target_records_with_images
    )

    return {
        "target_records_with_images": target_records_with_images,
        "api_call_target": effective_api_call_target,
        "soft_api_calls_per_hour": effective_soft_cap,
        "metadata_records_measured": metadata_records,
        "metadata_api_calls_measured": metadata_api_calls,
        "vision_images_measured": vision_images,
        "observed_records_per_api_call": records_per_api_call,
        "metadata_fetch_seconds_per_api_call": fetch_seconds_per_call,
        "vision_seconds_per_image": vision_seconds_per_image,
        "estimated_vision_seconds_for_100_images": vision_seconds_per_image * 100,
        "estimated_metadata_seconds_for_api_call_target": fetch_seconds_per_call * effective_api_call_target,
        "estimated_nonvision_pipeline_seconds_for_target_records": nonvision_record_seconds * target_records_with_images,
        "estimated_end_to_end_seconds_for_target_records_with_images": end_to_end_seconds,
        "api_calls_needed_for_target_records_at_observed_yield": api_calls_needed,
        "records_expected_at_api_call_target_observed_yield": effective_api_call_target * records_per_api_call,
        "exceeds_single_soft_cap_at_observed_yield": math.ceil(api_calls_needed) > effective_soft_cap,
    }


def _nonvision_record_seconds(timings: dict[str, Any], measured_records: int) -> float:
    per_record_steps = (
        "bronze_flattening_dedup",
        "silver_candidate_build",
        "dwc_mapping",
        "artifact_write",
    )
    return sum(float(timings.get(step, 0.0)) for step in per_record_steps) / measured_records


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None
