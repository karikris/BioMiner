from __future__ import annotations

import importlib
import json
from pathlib import Path

import polars as pl
import pytest


def _prepare(inputs, output_dir, **kwargs):
    try:
        module = importlib.import_module("biominer.embeddinggemma.prepare")
    except ModuleNotFoundError:
        pytest.fail("EmbeddingGemma input preparation is not implemented")
    return module.prepare(inputs, output_dir, **kwargs)


def _row(photo_id: str, first=(), second=(), **extra):
    def reviews(labels):
        return json.dumps(
            [
                {
                    "result": label,
                    "label": label.split(":", 1)[0],
                    "reviewer_uuid": "private-reviewer-token",
                    "identifiedBy": "private-person",
                }
                for label in labels
            ]
        )

    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "image_url": f"https://images.example/{photo_id}.jpg",
        "first_round_reviews_json": reviews(first),
        "first_round_review_count": len(first),
        "second_round_reviews_json": reviews(second),
        "second_round_review_count": len(second),
        **extra,
    }


def _input(tmp_path: Path, rows, name="source.parquet"):
    path = tmp_path / name
    pl.from_dicts(rows, infer_schema_length=None).write_parquet(path)
    return path


def _items(out):
    return pl.read_parquet(out / "items.parquet")


def test_explicit_benchmark_selection_preserves_subsets_without_automatic_sampling(
    tmp_path,
):
    path = _input(
        tmp_path,
        [
            _row("1", ["adult_butterfly"], image_sha256="a" * 64),
            _row("2", ["adult_butterfly"], image_sha256="b" * 64),
            _row("3", ["plant"], image_sha256="c" * 64),
            _row("4", ["no_biological_subject"], image_sha256="d" * 64),
        ],
    )
    _prepare([path], tmp_path / "baseline", sample_per_group=0)
    ids = dict(
        _items(tmp_path / "baseline").select("source_record_id", "item_id").iter_rows()
    )
    plan = tmp_path / "selection.parquet"
    pl.DataFrame(
        {
            "item_id": [ids["1"], ids["2"], ids["3"]],
            "benchmark_group": [
                "recent_butterfly",
                "earlier_butterfly",
                "other_categories",
            ],
        }
    ).write_parquet(plan)

    receipt = _prepare([path], tmp_path / "selected", sample_selection=plan)

    selected = _items(tmp_path / "selected").filter(pl.col("cache_group").is_not_null())
    assert selected.height == 3
    assert set(selected["source_record_id"]) == {"1", "2", "3"}
    assert receipt["cache_sampling"]["benchmark_groups"] == {
        "recent_butterfly": 1,
        "earlier_butterfly": 1,
        "other_categories": 1,
    }
    assert (
        pl.read_parquet(tmp_path / "selected" / "benchmark-selection.parquet").height
        == 3
    )


def test_explicit_benchmark_selection_rejects_duplicate_image_content(tmp_path):
    path = _input(
        tmp_path,
        [
            _row("1", ["adult_butterfly"], image_sha256="a" * 64),
            _row("2", ["adult_butterfly"], image_sha256="a" * 64),
        ],
    )
    _prepare([path], tmp_path / "baseline", sample_per_group=0)
    plan = tmp_path / "selection.parquet"
    pl.DataFrame(
        {
            "item_id": _items(tmp_path / "baseline")["item_id"],
            "benchmark_group": ["recent_butterfly", "earlier_butterfly"],
        }
    ).write_parquet(plan)

    with pytest.raises(ValueError, match="distinct known image content"):
        _prepare([path], tmp_path / "selected", sample_selection=plan)


def test_explicit_benchmark_selection_cannot_select_unreviewed_items(tmp_path):
    path = _input(tmp_path, [_row("1")])
    _prepare([path], tmp_path / "baseline", sample_per_group=0)
    plan = tmp_path / "selection.parquet"
    pl.DataFrame(
        {
            "item_id": _items(tmp_path / "baseline")["item_id"],
            "benchmark_group": ["recent_butterfly"],
        }
    ).write_parquet(plan)

    with pytest.raises(ValueError, match="decisive human"):
        _prepare([path], tmp_path / "selected", sample_selection=plan)


def test_prepare_deduplicates_work_and_retains_every_input_association(tmp_path):
    rows = [_row("1", ["adult_butterfly"]), _row("2")]
    a = _input(tmp_path, rows, "a.parquet")
    b = _input(tmp_path, [_row("1", ["adult_butterfly"])], "b.parquet")

    receipt = _prepare([b, a], tmp_path / "out", sample_per_group=2)

    items = _items(tmp_path / "out")
    associations = pl.read_parquet(tmp_path / "out" / "associations.parquet")
    assert items.height == 2
    assert associations.height == 3
    assert associations["item_id"].n_unique() == 2
    assert set(associations["input_path"]) == {str(a.resolve()), str(b.resolve())}
    assert receipt["counts"]["associations"] == 3
    assert receipt["counts"]["items"] == 2
    assert set(items["review_content_binding"]) == {"historical_content_unverified"}
    assert (
        receipt["evidence_maturity"]
        == "experimental_historical_review_not_occurrence_truth"
    )


def test_latest_round_dispute_never_falls_back_to_earlier_decisive_label(tmp_path):
    path = _input(tmp_path, [_row("1", ["adult_butterfly"], ["moth", "plant"])])

    _prepare([path], tmp_path / "out")

    row = _items(tmp_path / "out").row(0, named=True)
    assert row["review_status"] == "disputed"
    assert row["human_label"] is None
    assert row["human_group"] is None
    assert row["cache_group"] is None
    assert row["review_round"] == 2


def test_strict_majority_uses_all_votes_in_latest_reviewed_round(tmp_path):
    path = _input(tmp_path, [_row("1", ["plant"], ["moth", "moth", "adult_butterfly"])])

    _prepare([path], tmp_path / "out")

    row = _items(tmp_path / "out").row(0, named=True)
    assert row["review_status"] == "decisive"
    assert row["human_label"] == "moth"
    assert row["human_group"] == "other_biological"
    assert row["cache_group"] == "other_biological"


@pytest.mark.parametrize(
    ("labels", "status"),
    [
        ([], "unreviewed"),
        (["uncertain"], "uncertain"),
        (["image_unavailable"], "unavailable"),
    ],
)
def test_nondecisive_reviews_are_never_sampled(tmp_path, labels, status):
    path = _input(tmp_path, [_row("1", labels)])

    _prepare([path], tmp_path / "out")

    row = _items(tmp_path / "out").row(0, named=True)
    assert row["review_status"] == status
    assert row["cache_group"] is None


@pytest.mark.parametrize(
    ("label", "target", "expected"),
    [
        ("artifact_or_illustration", None, "non_biological"),
        ("no_biological_subject", None, "non_biological"),
        ("caterpillar", None, "other_biological"),
        ("target_scientific_name:Papilio demoleus", None, "butterfly"),
        ("target_scientific_name", "Papilio demoleus", "butterfly"),
        ("target_scientific_name:Quercus robur", None, None),
        ("target_scientific_name", None, None),
    ],
)
def test_group_mapping_requires_explicit_papilio_target_provenance(
    tmp_path, label, target, expected
):
    path = _input(tmp_path, [_row("1", [label], target_scientific_name=target)])

    _prepare([path], tmp_path / "out")

    assert _items(tmp_path / "out")["human_group"].to_list() == [expected]


def test_disagreeing_duplicate_identity_is_retained_as_disputed(tmp_path):
    a = _input(tmp_path, [_row("1", ["adult_butterfly"])], "a.parquet")
    b = _input(tmp_path, [_row("1", ["plant"])], "b.parquet")

    _prepare([a, b], tmp_path / "out")

    row = _items(tmp_path / "out").row(0, named=True)
    assert row["review_status"] == "disputed"
    assert row["review_conflict"] is True
    assert row["cache_group"] is None
    assert pl.read_parquet(tmp_path / "out" / "associations.parquet").height == 2


def test_sample_is_stable_across_input_row_order_and_reports_content_shortfall(
    tmp_path,
):
    duplicate_hash = "sha256:" + "a" * 64
    rows = [
        _row(str(i), ["adult_butterfly"], image_hash=duplicate_hash if i < 2 else None)
        for i in range(8)
    ]
    a = _input(tmp_path, rows, "a.parquet")
    b = _input(tmp_path, list(reversed(rows)), "b.parquet")

    receipt_a = _prepare([a], tmp_path / "a", seed="stable", sample_per_group=10)
    _prepare([b], tmp_path / "b", seed="stable", sample_per_group=10)

    chosen_a = _items(tmp_path / "a").filter(pl.col("cache_group").is_not_null())
    chosen_b = _items(tmp_path / "b").filter(pl.col("cache_group").is_not_null())
    assert chosen_a["item_id"].to_list() == chosen_b["item_id"].to_list()
    assert chosen_a.height == 7
    assert receipt_a["cache_sampling"]["groups"]["butterfly"]["shortfall"] == 3


def test_prepared_artifacts_exclude_reviewer_identity_and_comments(tmp_path):
    path = _input(
        tmp_path,
        [
            _row(
                "1",
                ["adult_butterfly"],
                reviewer_email="private-address@example.invalid",
                comment="private-comment",
            )
        ],
    )

    receipt = _prepare([path], tmp_path / "out")

    encoded = json.dumps(receipt)
    for name in ["items.parquet", "associations.parquet"]:
        frame = pl.read_parquet(tmp_path / "out" / name)
        encoded += json.dumps(frame.to_dicts())
        assert not any(
            "reviewer" in c or "identifiedby" in c.lower() for c in frame.columns
        )
    assert not any(
        secret in encoded
        for secret in [
            "private-reviewer-token",
            "private-person",
            "private-address",
            "private-comment",
        ]
    )


def test_same_inputs_reuse_verified_artifacts_without_rewriting(tmp_path):
    path = _input(tmp_path, [_row("1", ["adult_butterfly"])])
    output = tmp_path / "out"
    first = _prepare([path], output)
    before = {p.name: p.stat().st_mtime_ns for p in output.iterdir()}

    second = _prepare([path], output)

    assert second == first
    assert {p.name: p.stat().st_mtime_ns for p in output.iterdir()} == before


def test_existing_preparation_rejects_changed_input_bytes(tmp_path):
    path = _input(tmp_path, [_row("1", ["adult_butterfly"])])
    output = tmp_path / "out"
    _prepare([path], output)
    _input(tmp_path, [_row("1", ["plant"])])

    with pytest.raises(ValueError, match="fingerprint"):
        _prepare([path], output)


def test_existing_preparation_rejects_tampered_artifact(tmp_path):
    path = _input(tmp_path, [_row("1", ["adult_butterfly"])])
    output = tmp_path / "out"
    _prepare([path], output)
    (output / "items.parquet").write_bytes(b"not a parquet")

    with pytest.raises(ValueError, match="checksum"):
        _prepare([path], output)
