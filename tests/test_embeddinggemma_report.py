from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from biominer.embeddinggemma.common import atomic_json, file_sha256
from biominer.embeddinggemma.report import compare_configurations

LABELS = [
    {
        "score_index": 0,
        "label_id": "real_butterfly_closeup",
        "group": "Butterfly Subject",
        "text": "A real butterfly",
    },
    {
        "score_index": 1,
        "label_id": "real_butterfly_tiny",
        "group": "Butterfly Subject",
        "text": "A tiny real butterfly",
    },
    {
        "score_index": 2,
        "label_id": "butterfly_drawing",
        "group": "Butterfly Depictions",
        "text": "A drawing",
    },
]


def _configuration(
    tmp_path: Path,
    name: str,
    rows,
    *,
    runtime="runtime-same",
    source="source-same",
    cohort="cache",
    scores=None,
):
    path = tmp_path / name
    path.mkdir()
    pl.from_dicts(rows).write_parquet(path / "rankings.parquet")
    pl.from_dicts(LABELS).write_parquet(path / "labels.parquet")
    if scores is None:
        scores = [
            {"image_hash": row["image_hash"], "scores": [0.7, 0.2, 0.4]}
            for row in rows
            if row["status"] == "ok"
        ]
    pl.from_dicts(
        scores, schema={"image_hash": pl.String, "scores": pl.List(pl.Float32)}
    ).write_parquet(path / "scores.parquet")
    pl.DataFrame(
        {"embedding": [[1.0, 0.0]]}, schema={"embedding": pl.List(pl.Float32)}
    ).write_parquet(path / "image_embeddings.parquet")
    pl.DataFrame(
        {"embedding": [[1.0, 0.0]] * 3}, schema={"embedding": pl.List(pl.Float32)}
    ).write_parquet(path / "text_embeddings.parquet")
    atomic_json(
        path / "manifest.json",
        {
            "configuration": name,
            "budget": int(name.split("-")[0]),
            "dimension": int(name.split("-")[1]),
            "fingerprint": "score-" + name,
            "provenance": {
                "comparison_runtime_fingerprint": runtime,
                "input_manifest_sha256": source,
                "cohort": cohort,
            },
            "metrics": {
                "images_per_second": 2.5,
                "inference_seconds": 4.0,
                "peak_rss_bytes": 1024,
            },
            "outputs": {p.name: file_sha256(p) for p in path.glob("*.parquet")},
        },
    )
    return path


def _row(item_id="a", image_hash="hash-a", **changes):
    return {
        "item_id": item_id,
        "image_hash": image_hash,
        "status": "ok",
        "human_group": "butterfly",
        "review_status": "decisive",
        "leading_evidence_group": "butterfly",
        "top1_label": "real_butterfly_closeup",
        "top1_score": 0.7,
        "top1_top2_margin": 0.1,
        "flower_outranks_butterfly_on_flower": False,
        "real_vs_other_margin": 0.1,
        "real_vs_depiction_margin": 0.3,
        "review_content_binding": "historical_content_unverified",
        **changes,
    }


def test_pairs_require_same_item_and_same_content_hash(tmp_path):
    a = _configuration(tmp_path, "280-256", [_row("a", "before"), _row("b", "same")])
    b = _configuration(tmp_path, "1120-256", [_row("a", "after"), _row("b", "same")])

    result = compare_configurations([a, b], tmp_path / "report")

    pair = result["pairs"][0]
    assert pair["paired_scored"] == 1
    assert pair["content_hash_mismatches"] == 1
    assert pair["left_unmatched_scored"] == 1
    assert pair["right_unmatched_scored"] == 1
    paired = pl.read_parquet(tmp_path / "report" / pair["artifact"])
    assert paired["item_id"].to_list() == ["b"]


@pytest.mark.parametrize(
    "change",
    [{"runtime": "different"}, {"source": "different"}, {"cohort": "reviewed"}],
)
def test_incompatible_provenance_fails_closed(tmp_path, change):
    a = _configuration(tmp_path, "280-256", [_row()])
    b = _configuration(tmp_path, "1120-256", [_row()], **change)

    with pytest.raises(ValueError, match="compatible|cohort"):
        compare_configurations([a, b], tmp_path / "report")


def test_expected_dimension_and_budget_differences_are_allowed(tmp_path):
    a = _configuration(tmp_path, "280-256", [_row()])
    b = _configuration(tmp_path, "1120-768", [_row()])

    result = compare_configurations([a, b], tmp_path / "report")

    assert result["pairs"][0]["paired_scored"] == 1


def test_fully_unscored_corpus_has_explicit_coverage(tmp_path):
    a = _configuration(tmp_path, "280-256", [{"item_id": "a", "status": "unavailable"}])

    result = compare_configurations([a], tmp_path / "report")

    row = result["configurations"][0]
    assert row["scored"] == 0
    assert row["unscored"] == 1
    assert row["coverage"]["status_counts"] == {"unavailable": 1}
    assert row["groups"][0]["denominator"] == 0
    assert (tmp_path / "report" / "index.html").exists()


def test_subsets_report_item_and_scored_pair_denominators(tmp_path):
    a = _configuration(tmp_path, "280-256", [_row("a"), _row("b", "b")])
    b = _configuration(tmp_path, "1120-256", [_row("a"), _row("c", "c")])

    result = compare_configurations([a, b], tmp_path / "report")

    pair = result["pairs"][0]
    assert pair["same_item_cohort"] is False
    assert pair["left_only_items"] == 1
    assert pair["right_only_items"] == 1
    assert pair["paired_scored"] == 1


def test_report_includes_storage_metrics_confusion_and_axes(tmp_path):
    a = _configuration(
        tmp_path, "280-256", [_row(leading_evidence_group="non_biological")]
    )

    result = compare_configurations([a], tmp_path / "report")

    summary = result["configurations"][0]
    assert summary["storage"]["normalized_image_vector_bytes"] == 8
    assert summary["storage"]["normalized_text_vector_bytes"] == 24
    assert summary["metrics"]["images_per_second"] == 2.5
    confusion = pl.read_parquet(
        tmp_path / "report" / result["artifacts"]["group_confusion"]
    )
    assert confusion.select(
        "human_group", "leading_evidence_group", "count"
    ).to_dicts() == [
        {
            "human_group": "butterfly",
            "leading_evidence_group": "non_biological",
            "count": 1,
        }
    ]
    assert summary["real_vs_depiction"]["real_outranks_depiction"] == 1


def test_tiny_subject_gallery_ranks_raw_label_score_independently(tmp_path):
    rows = [_row("low", "low", top1_score=0.9), _row("high", "high", top1_score=0.1)]
    a = _configuration(
        tmp_path,
        "280-256",
        rows,
        scores=[
            {"image_hash": "low", "scores": [0.9, 0.1, 0.0]},
            {"image_hash": "high", "scores": [0.1, 0.8, 0.0]},
        ],
    )

    result = compare_configurations([a], tmp_path / "report")

    tiny = pl.read_parquet(
        tmp_path / "report" / result["artifacts"]["small_subject_candidates"]
    )
    assert tiny["item_id"].to_list() == ["high", "low"]
    html = (tmp_path / "report" / "index.html").read_text()
    assert "candidate evidence" in html
    assert "annotated tiny-subject performance" in html


def test_report_escapes_strings_and_never_embeds_remote_source_urls(tmp_path):
    a = _configuration(
        tmp_path,
        "280-256",
        [
            _row(
                top1_label="<script>alert(1)</script>",
                local_path="https://private.example/image?secret=private",
            )
        ],
    )

    compare_configurations([a], tmp_path / "report")

    html = (tmp_path / "report" / "index.html").read_text()
    assert "<script>" not in html
    assert "https://" not in html
    assert "private.example" not in html
    assert "scores are not probabilities" in html


def test_unscored_configuration_with_null_hashes_compares_to_scored_partner(tmp_path):
    a = _configuration(
        tmp_path,
        "280-256",
        [{"item_id": "a", "image_hash": None, "status": "unavailable"}],
    )
    b = _configuration(tmp_path, "1120-256", [_row()])

    result = compare_configurations([a, b], tmp_path / "report")

    assert result["pairs"][0]["paired_scored"] == 0
    assert result["pairs"][0]["right_unmatched_scored"] == 1
