from __future__ import annotations

import json

import pytest

from flickr_bio_occurrence.benchmark.estimates import estimate_combined_production, estimate_production_from_report


def test_estimate_production_from_live_report_scales_fetch_and_bioclip_timings() -> None:
    report = {
        "actual_unique_records": 53,
        "work_items_called": 100,
        "step_timings_seconds": {
            "flickr_fetch": 4.161630896,
            "bronze_flattening_dedup": 0.062556365,
            "silver_candidate_build": 0.002513293,
            "vision_classification": 274.721969564,
            "dwc_mapping": 0.001753343,
            "artifact_write": 0.063161644,
        },
        "api_policy": {
            "soft_api_calls_per_hour": 3200,
            "hard_api_calls_per_hour": 3600,
            "hard_photo_records_per_hour": 3600,
        },
    }

    estimate = estimate_production_from_report(report, target_records_with_images=3200, api_call_target=3200)

    assert estimate["measured_api_calls"] == 100
    assert estimate["measured_records_with_images"] == 53
    assert estimate["observed_records_per_api_call"] == pytest.approx(0.53)
    assert estimate["fetch_seconds_for_measured_api_calls"] == pytest.approx(4.161630896)
    assert estimate["bioclip_seconds_per_image"] == pytest.approx(274.721969564 / 53)
    assert estimate["estimated_bioclip_seconds_for_100_images"] == pytest.approx((274.721969564 / 53) * 100)
    assert estimate["estimated_metadata_seconds_for_api_call_target"] == pytest.approx(4.161630896 * 32)
    assert estimate["records_expected_at_api_call_target_observed_yield"] == pytest.approx(1696)
    assert estimate["api_calls_needed_for_target_records_at_observed_yield"] == pytest.approx(3200 / 0.53)
    assert estimate["exceeds_single_soft_cap_at_observed_yield"] is True
    assert estimate["estimated_end_to_end_seconds_for_target_records_with_images"] == pytest.approx(
        (4.161630896 / 100) * (3200 / 0.53)
        + ((0.062556365 + 0.002513293 + 0.001753343 + 0.063161644) / 53) * 3200
        + (274.721969564 / 53) * 3200
    )


def test_estimate_production_refuses_report_without_records() -> None:
    with pytest.raises(ValueError, match="actual_unique_records"):
        estimate_production_from_report({"actual_unique_records": 0, "work_items_called": 100}, target_records_with_images=3200)


def test_estimate_production_can_override_stale_report_soft_cap() -> None:
    report = {
        "actual_unique_records": 53,
        "work_items_called": 100,
        "step_timings_seconds": {"flickr_fetch": 4.0, "vision_classification": 265.0},
        "api_policy": {"soft_api_calls_per_hour": 3000},
    }

    estimate = estimate_production_from_report(
        report,
        target_records_with_images=3200,
        api_call_target=3200,
        soft_api_calls_per_hour=3200,
    )

    assert estimate["soft_api_calls_per_hour"] == 3200
    assert estimate["api_call_target"] == 3200


def test_qa_estimate_cli_outputs_production_estimate(tmp_path, capsys) -> None:
    from flickr_bio_occurrence.cli import build_parser, run

    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "actual_unique_records": 53,
                "work_items_called": 100,
                "step_timings_seconds": {"flickr_fetch": 4.0, "vision_classification": 265.0},
                "api_policy": {"soft_api_calls_per_hour": 3200},
            }
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "qa-estimate",
            "--report",
            str(report_path),
            "--target-records",
            "3200",
            "--api-call-target",
            "3200",
            "--soft-api-calls-per-hour",
            "3200",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["target_records_with_images"] == 3200
    assert payload["api_call_target"] == 3200
    assert payload["soft_api_calls_per_hour"] == 3200
    assert payload["estimated_bioclip_seconds_for_100_images"] == pytest.approx(500.0)


def test_estimate_combined_production_uses_metadata_and_vision_reports() -> None:
    metadata_report = {
        "actual_unique_records": 251,
        "work_items_called": 3100,
        "step_timings_seconds": {
            "flickr_fetch": 90.63428225900134,
            "bronze_flattening_dedup": 0.4822035050019622,
            "silver_candidate_build": 0.004604730995197315,
            "dwc_mapping": 0.006680452002910897,
            "artifact_write": 0.025397476005309727,
        },
    }
    vision_report = {
        "actual_unique_records": 100,
        "api_calls_made": 0,
        "step_timings_seconds": {"vision_classification": 496.36937554500037},
    }

    estimate = estimate_combined_production(
        metadata_report,
        vision_report,
        target_records_with_images=3200,
        api_call_target=3200,
        soft_api_calls_per_hour=3200,
    )

    assert estimate["metadata_records_measured"] == 251
    assert estimate["metadata_api_calls_measured"] == 3100
    assert estimate["vision_images_measured"] == 100
    assert estimate["observed_records_per_api_call"] == pytest.approx(251 / 3100)
    assert estimate["bioclip_seconds_per_image"] == pytest.approx(496.36937554500037 / 100)
    assert estimate["estimated_bioclip_seconds_for_100_images"] == pytest.approx(496.36937554500037)
    assert estimate["records_expected_at_api_call_target_observed_yield"] == pytest.approx(3200 * (251 / 3100))
    assert estimate["exceeds_single_soft_cap_at_observed_yield"] is True
    assert estimate["estimated_end_to_end_seconds_for_target_records_with_images"] == pytest.approx(
        (90.63428225900134 / 3100) * (3200 / (251 / 3100))
        + ((0.4822035050019622 + 0.004604730995197315 + 0.006680452002910897 + 0.025397476005309727) / 251) * 3200
        + (496.36937554500037 / 100) * 3200
    )


def test_qa_estimate_combined_cli_outputs_combined_estimate(tmp_path, capsys) -> None:
    from flickr_bio_occurrence.cli import build_parser, run

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "actual_unique_records": 10,
                "work_items_called": 20,
                "step_timings_seconds": {"flickr_fetch": 2.0},
            }
        ),
        encoding="utf-8",
    )
    vision_path = tmp_path / "vision.json"
    vision_path.write_text(
        json.dumps(
            {
                "actual_unique_records": 5,
                "api_calls_made": 0,
                "step_timings_seconds": {"vision_classification": 25.0},
            }
        ),
        encoding="utf-8",
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "qa-estimate-combined",
            "--metadata-report",
            str(metadata_path),
            "--vision-report",
            str(vision_path),
            "--target-records",
            "100",
            "--api-call-target",
            "3200",
            "--soft-api-calls-per-hour",
            "3200",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata_api_calls_measured"] == 20
    assert payload["vision_images_measured"] == 5
    assert payload["estimated_bioclip_seconds_for_100_images"] == pytest.approx(500.0)
