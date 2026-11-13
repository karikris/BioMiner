import numpy as np
import polars as pl
import pytest


def test_label_bank_preserves_groups_and_full_sentences(tmp_path):
    from biominer.embeddinggemma.labels import load_labels

    bank = tmp_path / "bank.md"
    bank.write_text(
        "## Butterfly Subject (1)\n\n- `real_butterfly_closeup` — A real butterfly\n"
        "## Plants Fungi (1)\n\n- `flower_closeup` — A flower closeup\n"
    )
    labels = load_labels(bank, expected_count=2)
    assert labels[0] == {
        "label_id": "real_butterfly_closeup",
        "group": "Butterfly Subject",
        "text": "A real butterfly",
    }


def test_duplicate_labels_are_rejected(tmp_path):
    from biominer.embeddinggemma.labels import load_labels

    bank = tmp_path / "bank.md"
    bank.write_text(
        "## Plants Fungi (2)\n- `flower` — A flower\n- `flower` — A flower\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_labels(bank, expected_count=2)


def fixture_vectors():
    labels = [
        {
            "label_id": "real_butterfly_on_flower",
            "group": "Butterfly Subject",
            "text": "A butterfly on a flower",
        },
        {"label_id": "flower_closeup", "group": "Plants Fungi", "text": "A flower"},
        {
            "label_id": "butterfly_cake",
            "group": "Butterfly Depictions",
            "text": "A butterfly cake",
        },
        {
            "label_id": "macro_photo",
            "group": "Visual Medium",
            "text": "A macro photograph",
        },
    ]
    vectors = np.zeros((4, 768), dtype=np.float32)
    vectors[0, 0] = 1
    vectors[1, 1] = 1
    vectors[2, 2] = 1
    vectors[3, 3] = 1
    return labels, vectors


def test_scores_keep_overlapping_labels_and_competing_axes(tmp_path):
    from biominer.embeddinggemma.scoring import score_configuration

    labels, text = fixture_vectors()
    vector = np.zeros(768, dtype=np.float32)
    vector[:4] = [0.3, 0.5, 0.1, 0.9]
    items = pl.DataFrame(
        {
            "item_id": ["a", "b"],
            "human_group": ["butterfly", None],
            "review_status": ["decisive", "unreviewed"],
            "cache_group": ["butterfly", None],
        }
    )
    media = pl.DataFrame(
        {
            "item_id": ["a", "b"],
            "status": ["ok", "unavailable"],
            "image_hash": ["hash-a", None],
            "local_path": ["image.png", None],
        }
    )
    embeddings = pl.DataFrame(
        {"image_hash": ["hash-a"], "embedding": [vector.tolist()]}
    )
    result = score_configuration(
        items,
        media,
        embeddings,
        labels,
        text,
        tmp_path,
        budget=280,
        dimension=256,
        provenance={"fingerprint": "fixture"},
    )
    rankings = pl.read_parquet(tmp_path / "rankings.parquet")
    valid = rankings.filter(pl.col("item_id") == "a").row(0, named=True)
    assert valid["top1_label"] == "flower_closeup"  # Medium remains a separate axis.
    assert valid["flower_outranks_butterfly_on_flower"] is True
    assert valid["real_vs_depiction_margin"] > 0
    assert rankings.filter(pl.col("item_id") == "b")["status"].item() == "unavailable"
    assert pl.read_parquet(tmp_path / "scores.parquet")["scores"].list.len().item() == 4
    assert result["scored"] == 1
    assert result["unscored"] == 1
    assert (
        pl.read_parquet(tmp_path / "image_embeddings.parquet")["embedding"]
        .list.len()
        .item()
        == 256
    )
    assert result["metrics"]["raw_image_embedding_bytes"] == 256 * 4


def test_all_unavailable_images_retain_typed_ranking_columns(tmp_path):
    from biominer.embeddinggemma.scoring import score_configuration

    labels, text = fixture_vectors()
    items = pl.DataFrame({"item_id": ["missing"], "human_group": ["butterfly"]})
    media = pl.DataFrame({"item_id": ["missing"], "status": ["unavailable"]})
    embeddings = pl.DataFrame(
        schema={"image_hash": pl.String, "embedding": pl.List(pl.Float32)}
    )
    score_configuration(
        items,
        media,
        embeddings,
        labels,
        text,
        tmp_path,
        budget=280,
        dimension=256,
        provenance={},
    )
    rankings = pl.read_parquet(tmp_path / "rankings.parquet")
    assert rankings["leading_evidence_group"].item() is None
    assert rankings["status"].item() == "unavailable"


@pytest.mark.parametrize(
    "real,other,prediction,status",
    [
        (0.8, 0.2, "butterfly", "scored"),
        (0.2, 0.8, "not_butterfly", "scored"),
        (0.5, 0.5, None, "ambiguous"),
    ],
)
def test_binary_projection_uses_existing_margin_and_preserves_ties(
    tmp_path, real, other, prediction, status
):
    from biominer.embeddinggemma.common import file_sha256
    from biominer.embeddinggemma.scoring import score_configuration

    labels, text = fixture_vectors()
    vector = np.zeros(768, dtype=np.float32)
    vector[:4] = [real, other, 0.1, 0.9]
    items = pl.DataFrame(
        {
            "item_id": ["image"],
            "human_label": ["adult_butterfly"],
            "human_group": ["butterfly"],
            "review_status": ["decisive"],
        }
    )
    media = pl.DataFrame(
        {"item_id": ["image"], "status": ["ok"], "image_hash": ["hash"]}
    )
    embeddings = pl.DataFrame({"image_hash": ["hash"], "embedding": [vector.tolist()]})
    manifest = score_configuration(
        items,
        media,
        embeddings,
        labels,
        text,
        tmp_path,
        budget=560,
        dimension=256,
        provenance={},
        binary=True,
    )
    path = tmp_path / "butterfly_classification.parquet"
    row = pl.read_parquet(path).row(0, named=True)
    assert row["prediction"] == prediction
    assert row["classification_status"] == status
    assert row["human_disagreement"] == (
        None if prediction is None else prediction != "butterfly"
    )
    assert row["real_vs_other_margin"] is not None
    assert manifest["outputs"][path.name] == file_sha256(path)


def test_binary_unscored_media_never_becomes_negative(tmp_path):
    from biominer.embeddinggemma.scoring import score_configuration

    labels, text = fixture_vectors()
    items = pl.DataFrame({"item_id": ["missing"]})
    media = pl.DataFrame({"item_id": ["missing"], "status": ["unavailable"]})
    empty = pl.DataFrame(
        schema={"image_hash": pl.String, "embedding": pl.List(pl.Float32)}
    )
    score_configuration(
        items,
        media,
        empty,
        labels,
        text,
        tmp_path,
        budget=560,
        dimension=768,
        provenance={},
        binary=True,
    )
    row = pl.read_parquet(tmp_path / "butterfly_classification.parquet").row(
        0, named=True
    )
    assert row["prediction"] is None
    assert row["butterfly_present"] is None
    assert row["classification_status"] == "unscored"


def test_changed_scoring_provenance_requires_new_output_directory(tmp_path):
    from biominer.embeddinggemma.common import file_sha256
    from biominer.embeddinggemma.scoring import score_configuration

    labels, text = fixture_vectors()
    items = pl.DataFrame({"item_id": ["missing"]})
    media = pl.DataFrame({"item_id": ["missing"], "status": ["unavailable"]})
    embeddings = pl.DataFrame(
        schema={"image_hash": pl.String, "embedding": pl.List(pl.Float32)}
    )
    score_configuration(
        items,
        media,
        embeddings,
        labels,
        text,
        tmp_path,
        budget=280,
        dimension=256,
        provenance={"input_manifest_sha256": "first"},
    )
    checksum = file_sha256(tmp_path / "manifest.json")
    with pytest.raises(ValueError, match="fingerprint"):
        score_configuration(
            items,
            media,
            embeddings,
            labels,
            text,
            tmp_path,
            budget=280,
            dimension=256,
            provenance={"input_manifest_sha256": "changed"},
        )
    assert file_sha256(tmp_path / "manifest.json") == checksum


def test_newly_encoded_images_change_scoring_identity(tmp_path):
    from biominer.embeddinggemma.scoring import score_configuration

    labels, text = fixture_vectors()
    items = pl.DataFrame({"item_id": ["a"]})
    media = pl.DataFrame({"item_id": ["a"], "status": ["ok"], "image_hash": ["hash"]})
    empty = pl.DataFrame(
        schema={"image_hash": pl.String, "embedding": pl.List(pl.Float32)}
    )
    score_configuration(
        items,
        media,
        empty,
        labels,
        text,
        tmp_path,
        budget=280,
        dimension=256,
        provenance={},
    )
    vectors = pl.DataFrame({"image_hash": ["hash"], "embedding": [[1.0] * 768]})
    with pytest.raises(ValueError, match="fingerprint"):
        score_configuration(
            items,
            media,
            vectors,
            labels,
            text,
            tmp_path,
            budget=280,
            dimension=256,
            provenance={},
        )


def test_paired_report_excludes_unscored_pairs(tmp_path):
    from biominer.embeddinggemma.report import compare_configurations

    for name in ("280-256", "1120-256"):
        path = tmp_path / name
        path.mkdir()
        from biominer.embeddinggemma.common import atomic_json

        atomic_json(
            path / "manifest.json",
            {
                "configuration": name,
                "scored": 1,
                "unscored": 1,
                "budget": int(name.split("-")[0]),
                "dimension": 256,
                "evidence_maturity": "experimental",
                "metrics": {},
                "fingerprint": name,
                "provenance": {
                    "comparison_runtime_fingerprint": "runtime",
                    "input_manifest_sha256": "source",
                    "cohort": "fixture",
                },
            },
        )
        labels, _ = fixture_vectors()
        pl.DataFrame(labels).with_row_index("score_index").write_parquet(
            path / "labels.parquet"
        )
        pl.DataFrame(
            {
                "item_id": ["a", "b"],
                "status": ["ok", "unavailable"],
                "image_hash": ["hash-a", None],
                "top1_label": ["flower_closeup", None],
                "human_group": ["butterfly", None],
                "leading_evidence_group": ["other_biological", None],
                "real_vs_other_margin": [0.1, None],
                "top1_score": [0.7, None],
                "top1_top2_margin": [0.05, None],
                "flower_outranks_butterfly_on_flower": [True, None],
            }
        ).write_parquet(path / "rankings.parquet")
    result = compare_configurations(
        [tmp_path / "280-256", tmp_path / "1120-256"], tmp_path / "report"
    )
    assert result["pairs"][0]["paired_scored"] == 1
    assert (tmp_path / "report" / "index.html").exists()
