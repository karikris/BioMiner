from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from biominer.benchmarks.prototype_benchmark_matrix import (
    EXPERIMENTS,
    NO_ACCURACY_REASON,
    PrototypeBenchmarkConfig,
    run_prototype_benchmark_matrix,
)


class FakeTextEmbedder:
    device = "mps"
    gpu_name = "Apple MPS fixture"
    worker_process_starts = 1

    def embed_text_labels(self, labels: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "Papilio demoleus" in label else [0.0, 1.0]
            for label in labels
        ]


def test_runs_all_b0_b16_rows_locally_and_quarantines_one_record(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path, skip_first=True)

    result = run_prototype_benchmark_matrix(
        config,
        text_embedder=FakeTextEmbedder(),
    )

    assert result.report["storage"] == {
        "backend": "local",
        "output_dir": str(config.output_dir),
        "s3_used": False,
    }
    assert result.report["counts"]["frozen_records"] == 81
    assert result.report["counts"]["records_scored"] == 80
    assert result.report["counts"]["records_skipped"] == 1
    assert result.skipped_path is not None
    skipped = pl.read_parquet(result.skipped_path)
    assert skipped["biological_negative"].to_list() == [False]
    assert skipped["skip_reason"].to_list() == ["fixture decode problem"]

    predictions = pl.read_parquet(result.predictions_path)
    summary = pl.read_parquet(result.experiment_summary_path)
    assert predictions.height == 80 * len(EXPERIMENTS)
    assert set(predictions["experiment_id"]) == {item[0] for item in EXPERIMENTS}
    assert predictions["classification_accuracy_permitted"].sum() == 0
    assert set(summary["classification_accuracy_status"]) == {NO_ACCURACY_REASON}
    assert summary["classification_accuracy"].null_count() == summary.height
    assert predictions.filter(pl.col("experiment_id") == "B11")[
        "availability_status"
    ].unique().to_list() == ["partial_raw_only_missing_focused_full_frame"]
    assert predictions.filter(pl.col("experiment_id") == "B12")[
        "availability_status"
    ].unique().to_list() == ["partial_raw_only_missing_focused_and_masked_full_frame"]


def test_rejects_s3_for_prototype_matrix(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path, skip_first=False)
    values = {field: getattr(config, field) for field in config.__dataclass_fields__}
    values["storage_backend"] = "s3"
    values["s3_permitted"] = True

    try:
        PrototypeBenchmarkConfig(**values)
    except ValueError as exc:
        assert "local-only storage" in str(exc)
    else:
        raise AssertionError("S3 configuration should be rejected")


def _fixture_config(
    tmp_path: Path,
    *,
    skip_first: bool,
) -> PrototypeBenchmarkConfig:
    embeddings_path = tmp_path / "embeddings.parquet"
    support_path = tmp_path / "support.parquet"
    candidates_path = tmp_path / "candidates.parquet"
    matrix_path = tmp_path / "matrix.json"
    rows = []
    splits = (
        ["support_train"] * 26
        + ["model_selection"] * 30
        + ["calibration"] * 13
        + ["final_test"] * 12
    )
    for index, split in enumerate(splits):
        target = index % 3 == 0
        rows.append(
            {
                "reference_media_id": f"media:{index:03d}",
                "accepted_taxon_key": "gbif:1938069" if target else "gbif:2",
                "scientific_name": (
                    "Papilio demoleus" if target else "Papilio fixture"
                ),
                "human_verified": False,
                "geo_cluster_id": ("geo:fixture" if index % 4 else "unassigned_geo"),
                "geographic_layer": "A" if index % 2 else "B",
                "route": "larval" if index == 80 else "adult_field",
                "trust_level": "R4",
                "dataset_split": split,
                "embedding_dimension": 2,
                "embedding": [1.0, 0.0] if target else [0.0, 1.0],
                "model_id": "imageomics/bioclip-2.5-vith14",
                "model_revision": "fixture-revision",
            }
        )
    pl.DataFrame(rows).with_columns(
        pl.col("embedding").cast(pl.Array(pl.Float32, 2)),
        pl.col("embedding_dimension").cast(pl.UInt32),
    ).write_parquet(embeddings_path)
    pl.DataFrame(
        {"reference_media_id": [row["reference_media_id"] for row in rows]}
    ).write_parquet(support_path)
    pl.DataFrame(
        {
            "accepted_taxon_key": ["gbif:1938069", "gbif:2"],
            "display_name": ["Papilio demoleus", "Papilio fixture"],
        }
    ).write_parquet(candidates_path)
    matrix_path.write_text(
        json.dumps(
            {"experiments": [{"experiment_id": f"B{index}"} for index in range(17)]}
        ),
        encoding="utf-8",
    )
    return PrototypeBenchmarkConfig(
        reference_embeddings=embeddings_path,
        reference_embeddings_sha256=_sha256(embeddings_path),
        support_manifest=support_path,
        support_manifest_sha256=_sha256(support_path),
        staged_candidate_scores=candidates_path,
        staged_candidate_scores_sha256=_sha256(candidates_path),
        experiment_matrix=matrix_path,
        experiment_matrix_sha256=_sha256(matrix_path),
        output_dir=tmp_path / "output",
        runtime_python=Path("/unused/python"),
        hf_cache_dir=tmp_path / "cache",
        model_name="imageomics/bioclip-2.5-vith14",
        model_revision="fixture-revision",
        open_clip_version="3.3.0",
        target_accepted_taxon_key="gbif:1938069",
        target_scientific_name="Papilio demoleus",
        skip_records=((("media:000", "fixture decode problem"),) if skip_first else ()),
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
