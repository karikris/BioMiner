"""Behavioral boundary checks for the ten-species suggestion contract."""

import json

import numpy as np
import polars as pl
import pytest

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.vision.custom_label_run import (
    RunState,
    embedding_key,
    export_results,
    score_vector,
)
from biominer.vision.species_results import historical_target_comparison


@pytest.mark.parametrize("status", ["pending", "media_failure"])
def test_unscored_images_preserve_all_ten_typed_null_rank_fields(tmp_path, status):
    keys = [("species", f"Papilio test{n:02d}") for n in range(12)]
    predictions = {} if status == "pending" else {"photo": {"bioclip_status": status}}
    export_results(
        [{"item_id": "image", "flickr_photo_id": "photo"}],
        predictions,
        {},
        tmp_path,
        keys,
        {"scoring_fingerprint": "ten-rank-unscored"},
        final=True,
    )
    frame = pl.read_parquet(tmp_path / "image_suggestions.parquet")
    assert frame["bioclip_status"].item() == status
    for rank in range(1, 11):
        name = f"bioclip_species_top{rank}"
        score = f"{name}_cosine"
        assert frame.schema[name] == pl.String
        assert frame.schema[score] == pl.Float64
        assert frame[name].item() is None
        assert frame[score].item() is None
    assert pl.read_parquet(tmp_path / "all_label_scores.parquet").height == 0


def test_ties_at_tenth_place_are_broken_by_species_name_independent_of_input_order():
    names = [f"Papilio test{n:02d}" for n in reversed(range(12))]
    result = score_vector(
        [1, 0], [("species", name) for name in names], np.array([[1, 0]] * 12)
    )
    assert [result[f"bioclip_species_top{rank}"] for rank in range(1, 11)] == sorted(
        names
    )[:10]
    assert result["bioclip_species_top10_cosine"] == pytest.approx(1)
    assert "bioclip_species_top11" not in result


def test_historical_target_at_rank_ten_counts_in_top_ten_only(tmp_path):
    target = "Papilio demoleus"
    row = {
        "human_label": f"target_scientific_name:{target}",
        "bioclip_status": "scored",
        "bioclip_species_top10": target,
    }
    comparison = historical_target_comparison([row], target=target)
    assert comparison["top1_matches"] == comparison["top5_matches"] == 0
    assert comparison["top10_matches"] == 1
    assert comparison["top10_match_fraction_of_scored"] == 1


def test_cli_top_ten_predictions_do_not_resume_legacy_top_five_payloads(
    tmp_path, monkeypatch
):
    from biominer.vision import custom_label_run as workflow

    config = {
        "version": "top-ten-test",
        "mode": "species",
        "templates": ["a photo of {description}"],
        "heads": {
            "species": {
                f"Papilio test{n:02d}": f"Papilio test{n:02d}" for n in range(12)
            }
        },
    }
    attestation = {"model_fingerprint": "model", "preprocessing_fingerprint": "pixels"}
    legacy_key = canonical_semantic_fingerprint(
        {
            "labels": config,
            "attestation": attestation,
            "scoring": "normalized-template-mean-cosine-v1",
        }
    )
    output = tmp_path / "output"
    output.mkdir()
    state = RunState(output / "run_state.sqlite")
    legacy = {"bioclip_status": "scored", "bioclip_species_top5": "Papilio test04"}
    state.save_prediction("photo", legacy_key, legacy)
    media_key = embedding_key({"pixel_sha256": "image"}, attestation)
    state.save_embedding(media_key, [1, 0])
    state.commit()
    state.connection.close()
    source = tmp_path / "source.parquet"
    pl.DataFrame(
        {
            "flickr_photo_id": ["photo"],
            "cached_image_path": [str(tmp_path / "source.image")],
            "cached_image_sha256": ["sha256:" + "a" * 64],
        }
    ).write_parquet(source)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(config))

    class Scorer:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    captured = []

    def run(
        rows,
        current,
        downloader,
        scorer,
        labels,
        directory,
        identity,
        manifest,
        **kwargs,
    ):
        captured.append(manifest)
        assert manifest["saved_species_ranks"] == 10
        assert manifest["scoring_fingerprint"] != legacy_key
        assert current.predictions(manifest["scoring_fingerprint"]) == {}
        assert current.predictions(legacy_key) == {"photo": legacy}
        np.testing.assert_array_equal(current.embedding(media_key), [1, 0])
        return {"counts": {"scored": 0}}

    monkeypatch.setattr(workflow, "PersistentBioClipScorer", Scorer)
    monkeypatch.setattr(workflow, "BioClipRuntime", lambda *args: object())
    monkeypatch.setattr(workflow, "attest", lambda scorer: attestation)
    monkeypatch.setattr(workflow, "run_inference", run)
    workflow.main(
        [
            "--source",
            str(source),
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--model-home",
            str(tmp_path),
            "--model-python",
            str(tmp_path / "python"),
            "--hf-cache",
            str(tmp_path),
            "--device",
            "cpu",
        ]
    )
    assert len(captured) == 1
