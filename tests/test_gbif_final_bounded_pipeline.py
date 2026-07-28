from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import biominer.gbif_final.bounded_pipeline as bounded_pipeline_module
from biominer.gbif_final.bounded_pipeline import (
    build_bounded_final_from_spine,
)
from biominer.gbif_final.spine import build_source_spine


def _write(
    path: Path,
    values: dict[str, list[object]],
    *,
    row_group_size: int = 2,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(values),
        path,
        compression="zstd",
        row_group_size=row_group_size,
    )
    return path


def test_bounded_pipeline_builds_and_resumes_exact_final_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = tmp_path / "inputs"
    pre_temporal = _write(
        inputs / "pre-temporal.parquet",
        {
            "gbifID": ["A", "X", "B"],
            "media_identifier": ["img-a", "img-x", "img-b"],
            "media_references": ["ref-a", "ref-x", "ref-b"],
            "speciesKey": ["1", "9", "2"],
            "species": ["Alpha", "Excluded", "Beta"],
        },
    )
    temporal = _write(
        inputs / "temporal.parquet",
        {
            "gbifID": ["A", "B"],
            "media_identifier": ["img-a", "img-b"],
            "media_references": ["ref-a", "ref-b"],
            "speciesKey": ["1", "2"],
            "species": ["Alpha", "Beta"],
            "derived_year": [2020, None],
            "derived_month": [5, None],
            "derived_day": [6, None],
            "temporal_derivation_method": ["event_date", None],
        },
        row_group_size=1,
    )
    media_quality = _write(
        inputs / "media-quality.parquet",
        {
            "source_row_id": ["source-0", "source-1", "source-2"],
            "source_sort_position": [0, 1, 2],
            "media_assertion_id": ["media-0", "media-1", "media-2"],
            "gbifID": ["A", "X", "B"],
            "media_check_status": ["pass", "pass", "review"],
        },
    )
    temporal_audit = _write(
        inputs / "temporal-audit.parquet",
        {
            "gbifID": ["X"],
            "temporal_derivation_status": ["excluded_pre_1960"],
            "source_media_rows": [1],
        },
    )
    spine = tmp_path / "source-spine"
    build_source_spine(
        temporal_parquet=temporal,
        pre_temporal_parquet=pre_temporal,
        media_quality_parquet=media_quality,
        temporal_audit_parquet=temporal_audit,
        output_directory=spine,
        producer_git_sha="deadbeef",
        part_rows=1,
        batch_rows=2,
    )

    occurrence_quality = _write(
        inputs / "occurrence-quality.parquet",
        {
            "gbifID": ["B", "A"],
            "occurrence_check_status": ["review", "pass"],
        },
    )
    rights_quality = _write(
        inputs / "rights-quality.parquet",
        {
            "source_row_id": ["source-0", "source-1", "source-2"],
            "media_assertion_id": ["media-0", "media-1", "media-2"],
            "gbifID": ["A", "X", "B"],
            "media_identifier": ["img-a", "img-x", "img-b"],
            "rights_policy_status": ["usable", "usable", "usable"],
        },
    )
    duplicate_quality = _write(
        inputs / "duplicate-quality.parquet",
        {
            "source_row_id": ["source-0", "source-1", "source-2"],
            "media_assertion_id": ["media-0", "media-1", "media-2"],
            "gbifID": ["A", "X", "B"],
            "duplicate_status": ["unique", "unique", "unique"],
        },
    )
    ai_one = _write(
        inputs / "ai-1.parquet",
        {
            "source_row_id": ["source-0", "source-1"],
            "media_assertion_id": ["media-0", "media-1"],
            "gbifID": ["A", "X"],
            "ai_readiness_status": ["ready", "ready"],
        },
    )
    ai_two = _write(
        inputs / "ai-2.parquet",
        {
            "source_row_id": ["source-2"],
            "media_assertion_id": ["media-2"],
            "gbifID": ["B"],
            "ai_readiness_status": ["review"],
        },
    )
    derived_dimension = _write(
        inputs / "derived-dimension.parquet",
        {
            "dimension_ordinal": [0],
            "gbifID": ["A"],
            "derived_quality_assertions": [
                [{"target_field": "continent", "derived_value": "Europe"}]
            ],
        },
    )
    species_dimension = _write(
        inputs / "species-dimension.parquet",
        {
            "dataset_species_key": ["1", "2"],
            "dataset_species": ["Alpha", "Beta"],
            "registry_match_status": ["matched", "matched"],
            "registry_match_method": ["key", "key"],
            "registry_taxon_key": ["taxon-1", "taxon-2"],
            "keyword_evidence": [["alpha"], ["beta"]],
            "keyword_source_assertions": [[], []],
            "flickr_query_terms": [["alpha"], ["beta"]],
        },
    )

    arguments = {
        "temporal_parquet": temporal,
        "source_spine_directory": spine,
        "media_quality_parquet": media_quality,
        "occurrence_quality_parquet": occurrence_quality,
        "rights_quality_parquet": rights_quality,
        "duplicate_quality_parquet": duplicate_quality,
        "ai_readiness_parts": [ai_two, ai_one],
        "derived_assertion_dimension": derived_dimension,
        "species_enrichment_dimension": species_dimension,
        "work_directory": tmp_path / "work",
        "output_directory": tmp_path / "final",
        "producer_git_sha": "deadbeef",
        "threads": 2,
        "memory_limit": "1GB",
        "batch_rows": 1,
        "final_row_group_size": 1,
        "free_space_multiplier": 1.0,
        "minimum_headroom_bytes": 0,
    }
    progress_events: list[dict[str, object]] = []
    arguments["progress"] = lambda event: progress_events.append(
        dict(event)
    )
    actual_materialize = (
        bounded_pipeline_module.seal_temporal_enriched_window
    )
    interrupted = False

    def stop_after_first_window(**kwargs: object) -> dict[str, object]:
        nonlocal interrupted
        receipt = actual_materialize(**kwargs)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return receipt

    monkeypatch.setattr(
        bounded_pipeline_module,
        "seal_temporal_enriched_window",
        stop_after_first_window,
    )
    with pytest.raises(KeyboardInterrupt):
        build_bounded_final_from_spine(**arguments)
    first_window = (
        tmp_path / "work" / "windows" / "part-00000" / "final.parquet"
    )
    first_window_mtime = first_window.stat().st_mtime_ns
    monkeypatch.setattr(
        bounded_pipeline_module,
        "seal_temporal_enriched_window",
        actual_materialize,
    )

    manifest = build_bounded_final_from_spine(**arguments)
    final_path = (
        tmp_path / "final" / "gbif_media_final_enriched.parquet"
    )
    first_mtime = final_path.stat().st_mtime_ns
    table = pq.read_table(final_path)

    assert manifest["counts"]["rows"] == 2
    assert first_window.stat().st_mtime_ns == first_window_mtime
    assert table["gbifID"].to_pylist() == ["A", "B"]
    assert table["source_row_id"].to_pylist() == [
        "source-0",
        "source-2",
    ]
    assert [
        value["media_check_status"]
        for value in table["media_quality"].to_pylist()
    ] == ["pass", "review"]
    assert table["derived_quality_assertions"].to_pylist() == [
        [{"derived_value": "Europe", "target_field": "continent"}],
        None,
    ]
    assert table["registry_match_status"].to_pylist() == [
        "matched",
        "matched",
    ]

    resumed = build_bounded_final_from_spine(**arguments)
    assert resumed == manifest
    assert final_path.stat().st_mtime_ns == first_mtime
    assert any(
        event["event"] == "partition_completed"
        and event["stage"] == "global_sidecar"
        and event["rows_passed"] == 2
        for event in progress_events
    )
    assert any(
        event["event"] == "partition_completed"
        and event["stage"] == "final_window"
        and event["checkpoint_path"].endswith(
            "final.parquet.receipt.json"
        )
        for event in progress_events
    )
    assert progress_events[-1]["event"] == "pipeline_reused"
    assert progress_events[-1]["rows_skipped_from_cache"] == 2
