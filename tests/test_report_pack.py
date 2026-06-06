from __future__ import annotations

import json
import importlib.util
from pathlib import Path


def _load_write_report_pack():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_report_pack.py"
    spec = importlib.util.spec_from_file_location("generate_report_pack", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.write_report_pack


write_report_pack = _load_write_report_pack()


REQUIRED_REPORTS = {
    "bioclip_run_summary.json",
    "bioclip_run_summary.md",
    "storage_profile.json",
    "cache_profile.json",
    "gpu_profile.json",
    "quality_profile.json",
    "idempotency_profile.json",
    "code_cleanup_report.md",
    "agents_update_recommendations.json",
}


def test_report_pack_generator_writes_required_reports(tmp_path) -> None:
    write_report_pack(tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == REQUIRED_REPORTS


def test_report_pack_records_comment_and_image_selection_status(tmp_path) -> None:
    write_report_pack(tmp_path)

    summary = json.loads((tmp_path / "bioclip_run_summary.json").read_text(encoding="utf-8"))

    assert summary["comment_handling"]["comments_currently_fetched"]["value"] is False
    assert summary["comment_handling"]["comments_stored_in_raw_payloads"]["value"] is False
    assert summary["comment_handling"]["comments_transformed_to_parquet"]["value"] is False
    assert summary["image_selection"]["order"] == ["url_l", "url_m"]
    assert summary["cli_commands"]["commands"] == [
        "fetch-live",
        "fetch-comments",
        "build-evidence",
        "classify-once",
        "classify-watch",
        "apply-rules",
        "gc-cache",
        "compact-parquet",
        "qa-summary",
    ]
    assert summary["evidence_first_pipeline"]["one_evidence_row_per_photo_record"] is True
    assert summary["evidence_first_pipeline"]["one_publication_state_per_record"] is True
    assert summary["evidence_first_pipeline"]["in_review_requires_review_reason"] is True
    assert summary["prediction_checkpoints"]["layout"] == (
        "silver/silver_vision_prediction/model_version=<model_id>/run_id=<run_id>/shard_id=<shard_id>/part-00000.parquet"
    )


def test_report_pack_marks_uninstrumented_metrics_explicitly(tmp_path) -> None:
    write_report_pack(tmp_path)

    gpu_profile = json.loads((tmp_path / "gpu_profile.json").read_text(encoding="utf-8"))
    quality_profile = json.loads((tmp_path / "quality_profile.json").read_text(encoding="utf-8"))

    assert gpu_profile["measured_gpu_name"] == "not_instrumented"
    assert gpu_profile["long_lived_service"]["symbol"] == "BioClipJobService"
    assert quality_profile["manual_ground_truth_accuracy"] is None
