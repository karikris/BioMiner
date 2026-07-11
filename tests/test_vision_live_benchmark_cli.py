from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from biominer.benchmarks.vision_plumbing import write_benchmark_taxonomy_store
from biominer.cli import build_parser, run


def test_dev_vision_benchmark_live_m5pro_parses_m5pro_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "dev",
            "vision",
            "benchmark-live-m5pro",
            "--input",
            "runs/local_debug/papilio_demoleus/canonical_source_records.parquet",
            "--taxonomy-candidate-table",
            "data/registry/current",
            "--taxonomy-text-embedding-cache",
            "data/registry/current/taxonomy_text_embeddings.parquet",
            "--output-dir",
            "reports/vision_benchmarks/m5pro_live",
        ]
    )

    assert args.vision_command == "benchmark-live-m5pro"
    assert args.dev_command == "vision"
    assert args.device == "mps"
    assert args.checkpoint == "yoloe-26s-seg.pt"
    assert args.yolo_sidecar_transport == "json_b64"
    assert args.imgsz == 768
    assert args.yolo_batch == 16
    assert args.bioclip_batch == 24
    assert args.crop_target_px == 336
    assert args.crop_padding_ratio == 0.08
    assert args.limit == 100
    assert args.taxonomy_text_embedding_cache == (
        "data/registry/current/taxonomy_text_embeddings.parquet"
    )


def test_dev_vision_benchmark_live_m5pro_reports_missing_runtime_paths(tmp_path, capsys) -> None:
    input_path = _benchmark_input(tmp_path)
    parser = build_parser()
    args = parser.parse_args(
        [
            "dev",
            "vision",
            "benchmark-live-m5pro",
            "--input",
            str(input_path),
            "--taxonomy-candidate-table",
            str(tmp_path / "taxonomy_store"),
            "--taxonomy-text-embedding-cache",
            str(tmp_path / "taxonomy_text_embeddings.parquet"),
            "--vision-runtime-python",
            str(tmp_path / "missing-yolo" / "python"),
            "--bioclip-runtime-python",
            str(tmp_path / "missing-bioclip" / "python"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert run(args) == 2

    payload = json.loads(capsys.readouterr().out)
    missing_fields = {item["field"] for item in payload["missing"]}
    assert payload["error"] == "missing_required_path"
    assert missing_fields == {"vision_runtime_python", "bioclip_runtime_python"}


def test_dev_vision_benchmark_live_m5pro_reports_missing_taxonomy_table(tmp_path, capsys) -> None:
    input_path = _benchmark_input(tmp_path)
    yolo_python = _fake_python(tmp_path / "yolo" / "bin" / "python")
    bioclip_python = _fake_python(tmp_path / "bioclip" / "bin" / "python")
    parser = build_parser()
    args = parser.parse_args(
        [
            "dev",
            "vision",
            "benchmark-live-m5pro",
            "--input",
            str(input_path),
            "--taxonomy-candidate-table",
            str(tmp_path / "missing_taxonomy_store"),
            "--taxonomy-text-embedding-cache",
            str(tmp_path / "taxonomy_text_embeddings.parquet"),
            "--vision-runtime-python",
            str(yolo_python),
            "--bioclip-runtime-python",
            str(bioclip_python),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert run(args) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "missing_taxonomy_candidate_table"
    assert payload["benchmark_kind"] == "vision_live_m5pro"


def test_dev_vision_benchmark_live_m5pro_requires_v3_embedding_cache(tmp_path, capsys) -> None:
    input_path = _benchmark_input(tmp_path)
    taxonomy_store = tmp_path / "taxonomy_store"
    write_benchmark_taxonomy_store(taxonomy_store)
    yolo_python = _fake_python(tmp_path / "yolo" / "bin" / "python")
    bioclip_python = _fake_python(tmp_path / "bioclip" / "bin" / "python")
    missing_cache = tmp_path / "missing_taxonomy_text_embeddings.parquet"
    parser = build_parser()
    args = parser.parse_args(
        [
            "dev",
            "vision",
            "benchmark-live-m5pro",
            "--input",
            str(input_path),
            "--taxonomy-candidate-table",
            str(taxonomy_store),
            "--taxonomy-text-embedding-cache",
            str(missing_cache),
            "--vision-runtime-python",
            str(yolo_python),
            "--bioclip-runtime-python",
            str(bioclip_python),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert run(args) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "benchmark_kind": "vision_live_m5pro",
        "error": "missing_taxonomy_text_embedding_cache",
        "taxonomy_text_embedding_cache": str(missing_cache),
    }


def _benchmark_input(tmp_path: Path) -> Path:
    input_path = tmp_path / "canonical_source_records.parquet"
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "bench-live-1",
                "image_url": "memory://bench-live-1",
            }
        ]
    ).write_parquet(input_path)
    return input_path


def _fake_python(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# fake python", encoding="utf-8")
    return path
