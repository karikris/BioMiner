import hashlib
import json

import numpy as np
import polars as pl
import pytest
from PIL import Image

from biominer.vision.custom_label_run import (
    MediaDownloader,
    RunState,
    embedding_key,
    export_results,
    load_labels,
    normalize_vectors,
    run_inference,
    score_vector,
)


@pytest.fixture
def species_labels():
    names = [
        "Papilio demoleus",
        "Papilio aegeus",
        "Danaus plexippus",
        "Pieris rapae",
        "Vanessa kershawi",
        "Junonia villida",
    ]
    return {
        "version": "species-test-v1",
        "mode": "species",
        "historical_comparison_target": "Papilio demoleus",
        "templates": ["a photo of {description}"],
        "heads": {"species": {name: name for name in names}},
    }


def test_species_config_accepts_one_scientific_name_head(tmp_path, species_labels):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(species_labels))
    assert load_labels(path) == species_labels


@pytest.mark.parametrize(
    "change",
    ["extra_head", "alias", "two_names", "missing_version", "missing_templates"],
)
def test_species_config_rejects_non_species_contract(tmp_path, species_labels, change):
    if change == "extra_head":
        species_labels["heads"]["context"] = {"a": "a", "b": "b", "c": "c"}
    elif change == "alias":
        species_labels["heads"]["species"]["Papilio demoleus"] = "lime butterfly"
    elif change == "two_names":
        species_labels["heads"]["species"] = {
            "Papilio demoleus": "Papilio demoleus",
            "Pieris rapae": "Pieris rapae",
        }
    else:
        del species_labels[change.removeprefix("missing_")]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(species_labels))
    with pytest.raises(ValueError):
        load_labels(path)


def test_species_ranks_five_raw_cosines_with_deterministic_name_ties(species_labels):
    names = list(species_labels["heads"]["species"])
    keys = [("species", name) for name in names]
    result = score_vector(
        [1, 0],
        keys,
        normalize_vectors([[1, 0], [1, 0], [0, 1], [-1, 0], [0, 1], [0, -1]]),
    )
    assert result["bioclip_species_top1"] == "Papilio aegeus"
    assert result["bioclip_species_top2"] == "Papilio demoleus"
    assert result["bioclip_species_top5"] == "Vanessa kershawi"
    assert result["bioclip_species_margin"] == pytest.approx(0)
    assert result["bioclip_species_cosine__Pieris rapae"] == pytest.approx(-1)
    assert (
        len([key for key in result if key.startswith("bioclip_species_cosine__")]) == 6
    )
    assert not any(
        "probab" in key or "subject" in key or "context" in key for key in result
    )
    assert result["bioclip_wild_status"] == "unresolved"


def test_species_three_candidates_keeps_top_five_schema(species_labels):
    keys = [("species", name) for name in list(species_labels["heads"]["species"])[:3]]
    result = score_vector([1, 0], keys, normalize_vectors([[1, 0], [0, 1], [-1, 0]]))
    assert result["bioclip_species_top4"] is None
    assert result["bioclip_species_top5_cosine"] is None


def test_species_top_ten_are_sorted_and_exported_with_scores(tmp_path):
    names = [f"Papilio test{n:02d}" for n in range(12)]
    keys = [("species", name) for name in names]
    angles = np.linspace(0.1, 2.5, 12)
    prototypes = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    result = score_vector([1, 0], keys, prototypes)
    assert [result[f"bioclip_species_top{rank}"] for rank in range(1, 11)] == names[:10]
    assert result["bioclip_species_top10_cosine"] == pytest.approx(np.cos(angles[9]))
    assert "bioclip_species_top11" not in result
    export_results(
        [{"item_id": "sample", "flickr_photo_id": "sample"}],
        {"sample": result},
        {},
        tmp_path,
        keys,
        {"scoring_fingerprint": "top-ten-test"},
        final=True,
    )
    frame = pl.read_parquet(tmp_path / "image_suggestions.parquet")
    assert frame["bioclip_species_top10"].item() == names[9]
    assert frame["bioclip_species_top10_cosine"].item() == pytest.approx(
        np.cos(angles[9])
    )
    assert names[9] in (tmp_path / "report.html").read_text()
    assert pl.read_parquet(tmp_path / "all_label_scores.parquet").height == 12


def test_species_top_ten_pad_missing_candidates_with_typed_nulls(
    tmp_path, species_labels
):
    keys = [("species", name) for name in list(species_labels["heads"]["species"])[:3]]
    result = score_vector([1, 0], keys, normalize_vectors([[1, 0], [0, 1], [-1, 0]]))
    assert result["bioclip_species_top10"] is None
    export_results(
        [{"item_id": "a", "flickr_photo_id": "a"}],
        {"a": result},
        {},
        tmp_path,
        keys,
        {"scoring_fingerprint": "top-ten-padding"},
    )
    frame = pl.read_parquet(tmp_path / "image_suggestions.parquet")
    assert frame.schema["bioclip_species_top10"] == pl.String
    assert frame.schema["bioclip_species_top10_cosine"] == pl.Float64
    assert frame["bioclip_species_top10"].item() is None


def cached_row(path, photo_id="001"):
    return {
        "flickr_photo_id": photo_id,
        "cached_image_path": str(path.resolve()),
        "cached_image_sha256": "sha256:"
        + hashlib.sha256(path.read_bytes()).hexdigest(),
        "image_url": "https://invalid.example/image",
    }


def test_cached_image_is_read_only_verified_exif_full_frame(tmp_path):
    source = tmp_path / "source.jpg"
    image = Image.new("RGB", (80, 20), "red")
    exif = image.getexif()
    exif[274] = 6
    image.save(source, exif=exif)
    original = source.read_bytes()
    row = cached_row(source)
    downloader = MediaDownloader(
        tmp_path / "output", {}, opener=lambda *a, **k: pytest.fail("network used")
    )
    _, media = downloader.download(row)
    assert media["status"] == "ready"
    assert media["rendition"] == "verified_local_cache"
    assert media["byte_sha256"] == row["cached_image_sha256"].removeprefix("sha256:")
    assert media["original_path"] == str(source.resolve())
    assert source.read_bytes() == original
    with Image.open(media["canonical_path"]) as canonical:
        assert canonical.size == (20, 80)
        assert canonical.mode == "RGB"
    assert list((tmp_path / "output").glob("*.original")) == []


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "digest",
        "undecodable",
        "bytes",
        "pixels",
        "relative",
        "missing_digest",
    ],
)
def test_expected_cached_image_failure_never_uses_network(tmp_path, failure):
    path = tmp_path / "source.png"
    Image.new("RGB", (10, 10), "blue").save(path)
    row = cached_row(path)
    options = {}
    if failure == "missing":
        path.unlink()
    elif failure == "digest":
        row["cached_image_sha256"] = "sha256:" + "0" * 64
    elif failure == "undecodable":
        path.write_bytes(b"not image bytes")
        row = cached_row(path)
    elif failure == "bytes":
        options["max_bytes"] = 1
    elif failure == "pixels":
        options["max_pixels"] = 1
    elif failure == "relative":
        row["cached_image_path"] = "source.png"
    elif failure == "missing_digest":
        del row["cached_image_sha256"]
    downloader = MediaDownloader(
        tmp_path / "output",
        {"001": [("url_l", "https://invalid.example/fallback")]},
        opener=lambda *a, **k: pytest.fail("network used"),
        **options,
    )
    _, media = downloader.download(row)
    assert media["status"] == "media_failure"
    assert media["failure_kind"] == "cached_image_failure"
    assert media["errors"]
    assert "pixel_sha256" not in media


def test_species_checkpoint_keeps_all_pending_and_failure_rows_without_legacy_exports(
    tmp_path, species_labels
):
    keys = [("species", name) for name in species_labels["heads"]["species"]]
    rows = [
        {"item_id": "a", "flickr_photo_id": "1", "human_label": "historical"},
        {"item_id": "b", "flickr_photo_id": "2", "human_label": None},
    ]
    progress = export_results(
        rows,
        {"1": {"bioclip_status": "media_failure", "bioclip_error": "decode"}},
        {},
        tmp_path,
        keys,
        {"scoring_fingerprint": "species-test"},
    )
    frame = pl.read_parquet(tmp_path / "image_suggestions.parquet")
    assert frame["bioclip_status"].to_list() == ["media_failure", "pending"]
    assert "bioclip_species_top5_cosine" in frame.columns
    assert progress["counts"] == {"scored": 0, "media_failure": 1, "pending": 1}
    assert progress["complete"] is False
    assert progress["artifacts_finalized"] is False
    assert not list(tmp_path.glob("*.csv"))
    assert not list(tmp_path.glob("*.xlsx"))


def test_species_final_export_links_cached_media_and_measures_historical_target_matches(
    tmp_path, species_labels
):
    target = "Papilio demoleus"
    source = tmp_path / "cache image.png"
    Image.new("RGB", (4, 3), "blue").save(source)
    keys = [("species", name) for name in species_labels["heads"]["species"]]
    predictions = {
        "1": score_vector(
            [1, 0],
            keys,
            normalize_vectors([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [-1, 0]]),
        ),
        "2": score_vector(
            [1, 0],
            keys,
            normalize_vectors([[0, 1], [1, 0], [0, 1], [0, 1], [0, 1], [-1, 0]]),
        ),
        "3": {"bioclip_status": "media_failure", "bioclip_error": "missing"},
    }
    rows = [
        {
            "item_id": str(i),
            "flickr_photo_id": str(i),
            "human_label": f"target_scientific_name:{target}",
            "selection_reason": "human_adult",
            "human_group": "<script>alert(1)</script>",
            "cached_image_path": str(source),
            "source_collections": '["collection-a"]',
        }
        for i in (1, 2, 3)
    ]
    progress = export_results(
        rows,
        predictions,
        {"1": {"original_path": str(source), "canonical_path": str(source)}},
        tmp_path,
        keys,
        {"scoring_fingerprint": "species-test", "labels": species_labels},
        final=True,
    )
    assert progress["complete"] is True
    assert progress["artifacts_finalized"] is True
    assert progress["counts"] == {"scored": 2, "media_failure": 1, "pending": 0}
    summary = progress["historical_target_comparison"]
    assert summary["historically_labeled_rows"] == 3
    assert summary["scored_labeled_rows"] == 2
    assert summary["unscored_labeled_rows"] == 1
    assert summary["top1_matches"] == 1
    assert summary["top5_matches"] == 2
    assert summary["top1_match_fraction_of_scored"] == pytest.approx(0.5)
    scores = pl.read_parquet(tmp_path / "all_label_scores.parquet")
    assert scores.height == 12
    assert set(scores["label"]) == set(species_labels["heads"]["species"])
    assert set(scores["item_id"]) == {"1", "2"}
    html = (tmp_path / "report.html").read_text()
    assert source.as_uri() in html
    assert "not ground truth" in html
    assert "<script>alert(1)</script>" not in html
    assert "Papilio demoleus" in html
    assert "report.html" in progress["artifact_sha256"]
    assert not list(tmp_path.glob("*.csv"))
    assert not list(tmp_path.glob("*.xlsx"))


def test_species_empty_score_export_has_typed_long_schema(tmp_path, species_labels):
    keys = [("species", name) for name in species_labels["heads"]["species"]]
    rows = [{"item_id": "a", "flickr_photo_id": "1"}]
    progress = export_results(
        rows,
        {"1": {"bioclip_status": "media_failure", "bioclip_error": "missing"}},
        {},
        tmp_path,
        keys,
        {"scoring_fingerprint": "species-test"},
        final=True,
    )
    frame = pl.read_parquet(tmp_path / "all_label_scores.parquet")
    assert frame.height == 0
    assert frame.schema["cosine_similarity"] == pl.Float64
    assert (
        progress["historical_target_comparison"]["top1_match_fraction_of_scored"]
        is None
    )


def test_read_only_previous_state_reuses_only_exact_model_preprocessing_key(tmp_path):
    attestation = {
        "model_fingerprint": "model-a",
        "preprocessing_fingerprint": "full-frame-a",
    }
    media = {"pixel_sha256": "pixels"}
    key = embedding_key(media, attestation)
    previous_path = tmp_path / "previous.sqlite"
    previous = RunState(previous_path)
    previous.save_embedding(key, [1, 0, 0])
    previous.commit()
    previous.connection.close()
    before = previous_path.read_bytes()
    state = RunState(tmp_path / "current.sqlite", cached_paths=[previous_path])
    np.testing.assert_array_equal(state.embedding(key), [1, 0, 0])
    changed_key = embedding_key(
        media, dict(attestation, preprocessing_fingerprint="different")
    )
    assert state.embedding(changed_key) is None
    assert previous_path.read_bytes() == before


class SpeciesScorer:
    def __init__(self):
        self.cache_metrics = {}
        self.memory_metrics = {}
        self.image_calls = 0

    def embed_text_labels(self, prompts):
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]

    def embed_image_paths(self, paths):
        self.image_calls += len(paths)
        self.last_image_content_hashes = [path.stem for path in paths]
        return [[1, 0, 0]] * len(paths)


class SpeciesDownloader:
    def download(self, row):
        photo_id = str(row["flickr_photo_id"])
        return photo_id, {
            "status": "ready",
            "pixel_sha256": "pixels",
            "canonical_path": "pixels.png",
        }


def test_species_inference_reuses_cached_embeddings_and_keeps_representative_smoke(
    tmp_path, species_labels
):
    rows = [
        {
            "item_id": str(i),
            "flickr_photo_id": str(i),
            "selection_reason": "human_adult" if i < 40 else "unreviewed_four_way",
            "source_collections": '["collection-a"]' if i % 2 else '["collection-b"]',
        }
        for i in range(80)
    ]
    attestation = {"model_fingerprint": "m", "preprocessing_fingerprint": "p"}
    state, scorer = RunState(tmp_path / "state.sqlite"), SpeciesScorer()
    result = run_inference(
        rows,
        state,
        SpeciesDownloader(),
        scorer,
        species_labels,
        tmp_path,
        attestation,
        {"scoring_fingerprint": "species-inference"},
    )
    assert result["counts"] == {"scored": 80, "media_failure": 0, "pending": 0}
    assert scorer.image_calls == 1
    smoke = json.loads((tmp_path / "smoke_test.json").read_text())
    selected = {str(row["flickr_photo_id"]): row for row in rows}
    assert smoke["rows"] == 32
    assert len({selected[pid]["selection_reason"] for pid in smoke["photo_ids"]}) == 2
    assert len({selected[pid]["source_collections"] for pid in smoke["photo_ids"]}) == 2
    assert (
        smoke["selection_policy"]
        == "deterministic-round-robin-selection-reason-and-collection-v1"
    )
    assert not (tmp_path / "review_sample_200.csv").exists()
    run_inference(
        rows,
        state,
        SpeciesDownloader(),
        scorer,
        species_labels,
        tmp_path,
        attestation,
        {"scoring_fingerprint": "changed-species-prompts"},
    )
    assert scorer.image_calls == 1


def test_species_smoke_balances_selection_reasons_before_many_collection_strata():
    from biominer.vision.species_results import species_smoke_order

    rows = [
        {
            "flickr_photo_id": f"h{i}",
            "selection_reason": "human_adult",
            "source_collections": json.dumps([f"collection-{i}"]),
        }
        for i in range(40)
    ]
    rows += [
        {
            "flickr_photo_id": "u",
            "selection_reason": "unreviewed_four_way",
            "source_collections": '["unreviewed"]',
        }
    ]
    first = species_smoke_order(rows)[:32]
    assert {row["selection_reason"] for row in first} == {
        "human_adult",
        "unreviewed_four_way",
    }
    assert species_smoke_order(list(reversed(rows))) == species_smoke_order(rows)


@pytest.mark.parametrize("vector", [[1, 0], [float("nan"), 0, 0], [2, 0, 0]])
def test_species_inference_rejects_incompatible_cached_vectors(
    tmp_path, species_labels, vector
):
    state = RunState(tmp_path / "state.sqlite")
    attestation = {"model_fingerprint": "m", "preprocessing_fingerprint": "p"}
    state.save_embedding(embedding_key({"pixel_sha256": "pixels"}, attestation), vector)
    with pytest.raises(ValueError, match="invalid cached image embedding"):
        run_inference(
            [{"flickr_photo_id": "1"}],
            state,
            SpeciesDownloader(),
            SpeciesScorer(),
            species_labels,
            tmp_path,
            attestation,
            {"scoring_fingerprint": "invalid-cache"},
        )


def test_species_inference_uses_read_only_compatible_previous_state(
    tmp_path, species_labels
):
    attestation = {"model_fingerprint": "m", "preprocessing_fingerprint": "p"}
    previous_path = tmp_path / "previous.sqlite"
    previous = RunState(previous_path)
    previous.save_embedding(
        embedding_key({"pixel_sha256": "pixels"}, attestation), [1, 0, 0]
    )
    previous.commit()
    previous.connection.close()
    state, scorer = (
        RunState(tmp_path / "current.sqlite", cached_paths=[previous_path]),
        SpeciesScorer(),
    )
    progress = run_inference(
        [{"flickr_photo_id": "1"}],
        state,
        SpeciesDownloader(),
        scorer,
        species_labels,
        tmp_path,
        attestation,
        {"scoring_fingerprint": "read-only-cache"},
    )
    assert scorer.image_calls == 0
    assert progress["counters"]["embedding_cache_hits"] == 1
    assert progress["counters"]["encoded_images"] == 0


def test_species_pending_artifact_has_stable_typed_score_schema(
    tmp_path, species_labels
):
    keys = [("species", name) for name in species_labels["heads"]["species"]]
    export_results(
        [{"flickr_photo_id": "1"}],
        {},
        {},
        tmp_path,
        keys,
        {"scoring_fingerprint": "pending"},
    )
    frame = pl.read_parquet(tmp_path / "image_suggestions.parquet")
    assert frame.schema["bioclip_species_top1"] == pl.String
    assert frame.schema["bioclip_species_top1_cosine"] == pl.Float64
    for _, name in keys:
        column = f"bioclip_species_cosine__{name}"
        assert frame.schema[column] == pl.Float64
        assert frame[column].to_list() == [None]


def test_historical_comparison_uses_configured_species():
    from biominer.vision.species_results import historical_target_comparison

    target = "Danaus plexippus"
    rows = [
        {
            "human_label": f"target_scientific_name:{target}",
            "bioclip_status": "scored",
            "bioclip_species_top1": target,
        }
    ]
    comparison = historical_target_comparison(rows, target=target)
    assert comparison["target_scientific_name"] == target
    assert comparison["historically_labeled_rows"] == 1
    assert comparison["top1_matches"] == 1
    assert comparison["top5_matches"] == 1


def test_historical_comparison_without_target_is_explicitly_unavailable(
    tmp_path, species_labels
):
    keys = [("species", name) for name in species_labels["heads"]["species"]]
    rows = [
        {
            "flickr_photo_id": "1",
            "human_label": "target_scientific_name:Papilio demoleus",
        }
    ]
    progress = export_results(
        rows, {}, {}, tmp_path, keys, {"scoring_fingerprint": "no-target"}, final=True
    )
    comparison = progress["historical_target_comparison"]
    assert comparison["target_scientific_name"] is None
    assert comparison["human_label"] is None
    assert comparison["historically_labeled_rows"] == 0
    assert comparison["scored_labeled_rows"] == 0
    assert comparison["top1_match_fraction_of_scored"] is None
    assert "No historical comparison target configured" in comparison["interpretation"]
    assert (
        "No historical comparison target configured"
        in (tmp_path / "report.html").read_text()
    )
