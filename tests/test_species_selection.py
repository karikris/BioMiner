import polars as pl
import pytest


def test_exact_duplicate_names_preserve_first_occurrence():
    from biominer.vision.species_selection import species_labels

    config, audit = species_labels(
        ["Papilio aegeus", "Papilio demoleus", "Junonia lemonias", "Papilio aegeus"]
    )
    assert list(config["heads"]["species"]) == [
        "Papilio aegeus",
        "Papilio demoleus",
        "Junonia lemonias",
    ]
    assert audit["duplicates"] == [
        {"name": "Papilio aegeus", "position": 4, "first_position": 1}
    ]
    assert config["mode"] == "species"
    assert len(config["templates"]) == 3


def test_species_cli_rejects_missing_local_cache_before_model_start(
    tmp_path, monkeypatch
):
    import json

    from biominer.vision import custom_label_run
    from biominer.vision.species_selection import species_labels

    source = tmp_path / "source.parquet"
    pl.DataFrame(
        {"flickr_photo_id": ["one"], "image_url": ["https://example.test/one"]}
    ).write_parquet(source)
    config, _ = species_labels(
        ["Papilio aegeus", "Papilio demoleus", "Junonia lemonias"]
    )
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(config))
    monkeypatch.setattr(
        custom_label_run,
        "PersistentBioClipScorer",
        lambda **kwargs: pytest.fail("must not load model for missing local bytes"),
    )
    with pytest.raises(ValueError, match="local cache"):
        custom_label_run.main(
            [
                "--source",
                str(source),
                "--labels",
                str(labels),
                "--output",
                str(tmp_path / "run"),
                "--model-home",
                str(tmp_path),
                "--model-python",
                str(tmp_path / "python"),
                "--hf-cache",
                str(tmp_path / "hf"),
            ]
        )


def test_human_adults_override_model_but_reviewed_rejections_do_not():
    from biominer.vision.species_selection import select_candidates

    items = pl.DataFrame(
        {
            "item_id": ["adult", "target", "unseen", "three", "rejected", "disputed"],
            "human_label": [
                "adult_butterfly",
                "target_scientific_name:Papilio demoleus",
                None,
                None,
                "plant",
                None,
            ],
            "review_status": [
                "decisive",
                "decisive",
                "unreviewed",
                "unreviewed",
                "decisive",
                "disputed",
            ],
            "source_record_id": ["1", "2", "3", "4", "5", "6"],
            "source_provider": ["flickr"] * 6,
        }
    )
    predictions = {}
    for name in ("280-256", "1120-256", "280-768", "1120-768"):
        predictions[name] = pl.DataFrame(
            {
                "item_id": items["item_id"],
                "status": ["ok"] * 6,
                "image_hash": ["hash1", "hash2", "hash3", "hash4", "hash5", "hash6"],
                "leading_evidence_group": [
                    "other_biological",
                    "butterfly",
                    "butterfly",
                    "other_biological" if name == "280-768" else "butterfly",
                    "butterfly",
                    "butterfly",
                ],
            }
        )
    associations = pl.DataFrame(
        {
            "item_id": ["adult", "target", "unseen", "unseen"],
            "input_path": [
                "/original/a.parquet",
                "/original/a.parquet",
                "/original/a.parquet",
                "/geotagged/a.parquet",
            ],
        }
    )
    selected, links, counts = select_candidates(
        items, predictions, associations, target_scientific_name="Papilio demoleus"
    )
    assert selected["item_id"].to_list() == ["adult", "target", "unseen"]
    assert counts["human_adult"] == 2 and counts["unreviewed_four_way"] == 1
    assert links.height == 4
    assert (
        selected.filter(pl.col("item_id") == "unseen")["source_collections"].item()
        == '["geotagged", "original"]'
    )


def test_mismatched_content_across_configurations_fails_closed():
    from biominer.vision.species_selection import select_candidates

    items = pl.DataFrame(
        {
            "item_id": ["a"],
            "review_status": ["unreviewed"],
            "human_label": [None],
            "source_record_id": ["1"],
            "source_provider": ["flickr"],
        }
    )
    predictions = {
        name: pl.DataFrame(
            {
                "item_id": ["a"],
                "status": ["ok"],
                "image_hash": ["hash" if name != "280-768" else "changed"],
                "leading_evidence_group": ["butterfly"],
            }
        )
        for name in ("280-256", "1120-256", "280-768", "1120-768")
    }
    with pytest.raises(ValueError, match="content"):
        select_candidates(
            items,
            predictions,
            pl.DataFrame(schema={"item_id": pl.String, "input_path": pl.String}),
            target_scientific_name="Papilio demoleus",
        )


def test_prepared_selection_resumes_and_rejects_changed_names(tmp_path):
    import io
    import json

    import httpx
    from PIL import Image

    from biominer.embeddinggemma.common import atomic_json, file_sha256
    from biominer.embeddinggemma.media import cache_items
    from biominer.embeddinggemma.prepare import prepare
    from biominer.vision.species_selection import CONFIGS, prepare_selection

    source = tmp_path / "source.parquet"
    pl.DataFrame(
        {
            "source": ["flickr"],
            "flickr_photo_id": ["one"],
            "image_url": ["https://example.test/one.png"],
            "first_round_review_count": [1],
            "first_round_reviews_json": ['[{"result":"adult_butterfly"}]'],
        }
    ).write_parquet(source)
    benchmark = tmp_path / "benchmark"
    prepare([source], benchmark / "prepared")
    atomic_json(benchmark / "run.json", {"status": "complete"})
    items = pl.read_parquet(benchmark / "prepared/items.parquet")
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buffer, format="PNG")
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=buffer.getvalue())
        )
    ) as client:
        media = cache_items(items, tmp_path / "cache", http_client=client)
    for config in CONFIGS:
        path = benchmark / "scores/all" / config
        path.mkdir(parents=True)
        pl.DataFrame(
            {
                "item_id": items["item_id"],
                "image_hash": media["image_hash"],
                "status": ["ok"],
                "leading_evidence_group": ["butterfly"],
            }
        ).write_parquet(path / "rankings.parquet")
        atomic_json(
            path / "manifest.json",
            {
                "configuration": config,
                "outputs": {"rankings.parquet": file_sha256(path / "rankings.parquet")},
                "provenance": {
                    "input_manifest_sha256": file_sha256(
                        benchmark / "prepared/manifest.json"
                    )
                },
            },
        )
    names = ["Papilio aegeus", "Papilio demoleus", "Junonia lemonias"]
    output = tmp_path / "selection"
    first = prepare_selection(
        benchmark,
        tmp_path / "cache",
        names,
        output,
        target_scientific_name="Papilio demoleus",
    )
    assert (
        list(json.loads((output / "labels.json").read_text())["heads"]["species"])
        == names
    )
    assert first["counts"]["selected"] == 1
    assert (
        prepare_selection(
            benchmark,
            tmp_path / "cache",
            names,
            output,
            target_scientific_name="Papilio demoleus",
        )
        == first
    )
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_selection(
            benchmark,
            tmp_path / "cache",
            names + ["Papilio memnon"],
            output,
            target_scientific_name="Papilio demoleus",
        )
    selected = output / "selected_images.parquet"
    selected.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        prepare_selection(
            benchmark,
            tmp_path / "cache",
            names,
            output,
            target_scientific_name="Papilio demoleus",
        )
