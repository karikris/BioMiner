from __future__ import annotations

import inspect

from biominer.run.orchestrator import ProductionRunOrchestrator


def test_cloud_poll_stage_does_not_bridge_through_local_sqlite_or_tempdir() -> None:
    source = _method_source("_run_cloud_poll_flickr_stage")

    _assert_source_excludes(
        source,
        {
            "tempfile.TemporaryDirectory": "cloud poll workers must not create local worker databases",
            "MetadataPollState": "cloud poll workers must claim and canonicalize directly from WorkStore",
            "flickr_poller.sqlite": "production cloud mode must not create SQLite poll state",
            "canonical_source_records_frame": "cloud poll output must be per-work canonical delta shards, not local state snapshots",
        },
    )


def test_cloud_detection_stage_is_claimed_shard_driven_not_local_batch_file_driven() -> None:
    source = _cloud_branch_source("_run_detect_objects_stage")

    _assert_source_excludes(
        source,
        {
            "tempfile.TemporaryDirectory": "cloud detection must write bounded detection shards, not local temp final files",
            ".read_parquet(": "cloud detection must plan from shard inventory or scans, not eager-read full source-record outputs",
            ".to_dicts(": "cloud detection must claim bounded batches, not materialize all source records",
        },
    )


def test_cloud_bioclip_stage_is_claimed_shard_driven_not_local_batch_file_driven() -> None:
    source = _cloud_branch_source("_run_score_bioclip_stage")

    _assert_source_excludes(
        source,
        {
            "tempfile.TemporaryDirectory": "cloud BioCLIP scoring must write bounded score shards, not local temp final files",
            ".read_parquet(": "cloud BioCLIP scoring must read shard/batch inputs lazily or by claim, not eager-read full outputs",
            ".to_dicts(": "cloud BioCLIP scoring must build candidate context without materializing all records",
        },
    )


def test_cloud_join_summary_and_review_queue_are_shard_inventory_driven() -> None:
    for method_name in ("_run_join_evidence_stage", "_run_summarize_stage"):
        source = _cloud_branch_source(method_name)
        _assert_source_excludes(
            source,
            {
                ".read_parquet(": f"{method_name} must consume shard inventory/lazy scans, not full eager reads",
                "write_parquet_shard(plan.artifact_uris.object_evidence_uri": "cloud joins must write immutable output shards",
                "write_parquet_shard(plan.artifact_uris.review_queue_uri": "cloud review queues must be generated from summary shards",
            },
        )


def _method_source(method_name: str) -> str:
    return inspect.getsource(getattr(ProductionRunOrchestrator, method_name))


def _cloud_branch_source(method_name: str) -> str:
    source = _method_source(method_name)
    if "if is_cloud_uri(self.request.output_root):" not in source:
        return source
    cloud_branch = source.split("if is_cloud_uri(self.request.output_root):", 1)[1]
    local_markers = (
        "\n        missing = _missing_paths",
        "\n        if missing:",
        "\n        import json",
    )
    cut_points = [cloud_branch.find(marker) for marker in local_markers if cloud_branch.find(marker) > 0]
    return cloud_branch[: min(cut_points)] if cut_points else cloud_branch


def _assert_source_excludes(source: str, forbidden: dict[str, str]) -> None:
    hits = {pattern: reason for pattern, reason in forbidden.items() if pattern in source}
    assert not hits, "\n".join(f"{pattern}: {reason}" for pattern, reason in sorted(hits.items()))
