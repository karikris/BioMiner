from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from biominer.evaluation.holdouts import (
    BALANCED_CHALLENGE_CATEGORIES,
    BALANCED_CHALLENGE_HOLDOUT_FILE,
    FROZEN_EVALUATION_HOLDOUT_REPORT_FILE,
    FROZEN_EVALUATION_HOLDOUT_REPORT_MARKDOWN_FILE,
    FROZEN_EVALUATION_HOLDOUT_SCHEMA,
    NATURAL_STREAM_HOLDOUT_FILE,
    NATURAL_STREAM_SELECTION_FILE,
    NATURAL_STREAM_SELECTION_SCHEMA,
    FrozenHoldoutConfig,
    build_balanced_challenge_holdout,
    empty_frozen_evaluation_holdout,
    empty_natural_stream_selection,
    freeze_natural_stream_holdout,
    load_natural_stream_selection,
    publish_frozen_evaluation_holdouts,
    select_natural_stream_candidates,
    validate_evaluation_holdouts_disjoint,
    validate_frozen_evaluation_holdout,
    validate_natural_stream_selection,
    write_natural_stream_selection,
)
from biominer.evaluation.labels import (
    REVIEWED_LABEL_SCHEMA,
    REVIEWED_LABEL_SCHEMA_VERSION,
)
from biominer.evaluation.sampling import (
    EVALUATION_SAMPLING_FRAME_SCHEMA,
    EVALUATION_SAMPLING_FRAME_SCHEMA_VERSION,
)


TARGET = "gbif:target"


def test_empty_holdout_frames_have_stable_schemas() -> None:
    assert empty_natural_stream_selection().schema == NATURAL_STREAM_SELECTION_SCHEMA
    assert empty_frozen_evaluation_holdout().schema == FROZEN_EVALUATION_HOLDOUT_SCHEMA


def test_natural_selection_is_deterministic_and_preserves_prevalence_by_weight() -> (
    None
):
    sampling = _natural_sampling_frame()
    config = _config(natural_sample_size=4)

    first = select_natural_stream_candidates(sampling, config)
    second = select_natural_stream_candidates(sampling.reverse(), config)

    assert_frame_equal(first, second)
    assert first.schema == NATURAL_STREAM_SELECTION_SCHEMA
    assert first.height == 4
    assert set(first.columns).isdisjoint(
        {"target_present", "accepted_taxon_key", "scientific_name"}
    )
    by_domain = {
        json.loads(row["sampling_stratum_json"])["initial_visual_domain"]: row
        for row in first.group_by("sampling_stratum_id").first().to_dicts()
    }
    assert by_domain["live_field"]["population_stratum_size"] == 9
    assert by_domain["live_field"]["sample_stratum_size"] == 3
    assert by_domain["live_field"]["sampling_weight"] == pytest.approx(3.0)
    assert by_domain["artwork"]["population_stratum_size"] == 1
    assert by_domain["artwork"]["sample_stratum_size"] == 1
    assert by_domain["artwork"]["sampling_weight"] == pytest.approx(1.0)
    assert first["sampling_weight"].sum() == pytest.approx(10.0)
    validate_natural_stream_selection(first)


def test_natural_selection_excludes_forbidden_usage_before_weighting() -> None:
    sampling = _natural_sampling_frame()
    excluded_id = str(sampling["sampling_unit_id"][0])
    assignments = pl.DataFrame(
        [{"sampling_unit_id": excluded_id, "usage_role": "support_train"}]
    )

    selection = select_natural_stream_candidates(
        sampling,
        _config(natural_sample_size=4),
        usage_assignments=assignments,
    )

    assert excluded_id not in set(selection["sampling_unit_id"])
    assert selection["population_size"].unique().to_list() == [10]
    assert selection["eligible_population_size"].unique().to_list() == [9]
    assert selection["sampling_weight"].sum() == pytest.approx(9.0)


def test_natural_selection_rejects_tampered_stratum_identity() -> None:
    selection = select_natural_stream_candidates(
        _natural_sampling_frame(),
        _config(natural_sample_size=4),
    )
    stratum_id = str(selection["sampling_stratum_id"][0])
    tampered = selection.with_columns(
        pl.when(pl.col("sampling_stratum_id") == stratum_id)
        .then(pl.lit('{"initial_visual_domain":"tampered"}'))
        .otherwise(pl.col("sampling_stratum_json"))
        .alias("sampling_stratum_json")
    )

    with pytest.raises(ValueError, match="sampling_stratum_id is invalid"):
        validate_natural_stream_selection(tampered)


def test_natural_selection_parquet_is_immutable_and_round_trips(
    tmp_path: Path,
) -> None:
    selection = select_natural_stream_candidates(
        _natural_sampling_frame(),
        _config(natural_sample_size=4),
    )

    path = write_natural_stream_selection(selection, tmp_path / "selection")

    assert path.name == NATURAL_STREAM_SELECTION_FILE
    assert_frame_equal(load_natural_stream_selection(path), selection)
    with pytest.raises(FileExistsError):
        write_natural_stream_selection(selection, path)


def test_natural_selection_rejects_impossible_minimum_per_stratum() -> None:
    with pytest.raises(ValueError, match="minimum_per_natural_stratum"):
        select_natural_stream_candidates(
            _natural_sampling_frame(),
            _config(natural_sample_size=1, minimum_per_natural_stratum=1),
        )


def test_balanced_challenge_contains_every_required_class_equally() -> None:
    sampling, labels = _challenge_inputs(per_category=2)
    config = _config(challenge_per_category=1)

    first = build_balanced_challenge_holdout(sampling, labels, config)
    second = build_balanced_challenge_holdout(
        sampling.reverse(),
        labels.reverse(),
        config,
    )

    assert_frame_equal(first, second)
    assert first.schema == FROZEN_EVALUATION_HOLDOUT_SCHEMA
    assert first.height == len(BALANCED_CHALLENGE_CATEGORIES)
    assert first.group_by("evaluation_class").len()["len"].to_list() == [1] * 7
    assert set(first["evaluation_class"]) == set(BALANCED_CHALLENGE_CATEGORIES)
    assert first["sampling_weight"].null_count() == first.height
    assert set(first["dataset_split"]) == {"final_test"}
    validate_frozen_evaluation_holdout(first)


def test_balanced_challenge_reports_exact_category_shortfall() -> None:
    sampling, labels = _challenge_inputs(per_category=1)
    missing_id = str(
        labels.filter(pl.col("visual_domain") == "artwork")["flickr_photo_id"][0]
    )
    labels = labels.filter(pl.col("flickr_photo_id") != missing_id)

    with pytest.raises(ValueError, match='"artifacts"'):
        build_balanced_challenge_holdout(
            sampling,
            labels,
            _config(challenge_per_category=1),
        )


def test_balanced_challenge_rejects_support_or_training_overlap() -> None:
    sampling, labels = _challenge_inputs(per_category=1)
    sampling_unit_id = str(sampling["sampling_unit_id"][0])
    assignments = pl.DataFrame(
        [{"sampling_unit_id": sampling_unit_id, "usage_role": "calibration"}]
    )

    with pytest.raises(ValueError, match="forbidden evaluation usage"):
        build_balanced_challenge_holdout(
            sampling,
            labels,
            _config(challenge_per_category=1),
            usage_assignments=assignments,
        )


def test_natural_holdout_freezes_only_after_every_selected_item_is_reviewed() -> None:
    sampling = _natural_sampling_frame()
    config = _config(natural_sample_size=4)
    selection = select_natural_stream_candidates(sampling, config)
    labels = _labels_for_selection(sampling, selection)

    holdout = freeze_natural_stream_holdout(
        sampling,
        selection,
        labels,
        config,
    )

    assert holdout.height == 4
    assert holdout["sampling_weight"].sum() == pytest.approx(10.0)
    assert set(holdout["holdout_kind"]) == {"natural_stream"}
    assert set(holdout["dataset_split"]) == {"final_test"}
    validate_frozen_evaluation_holdout(holdout)

    incomplete = labels.head(labels.height - 1)
    with pytest.raises(ValueError, match="lacks completed final_test labels"):
        freeze_natural_stream_holdout(
            sampling,
            selection,
            incomplete,
            config,
        )


def test_natural_selection_cannot_depend_on_reviewed_outcomes() -> None:
    sampling = _natural_sampling_frame()
    config = _config(natural_sample_size=4)

    before_review = select_natural_stream_candidates(sampling, config)
    labels = _labels_for_selection(sampling, before_review)
    inverted = labels.with_columns((~pl.col("target_present")).alias("target_present"))
    after_unrelated_label_change = select_natural_stream_candidates(sampling, config)

    assert labels["target_present"].to_list() != inverted["target_present"].to_list()
    assert_frame_equal(before_review, after_unrelated_label_change)


def test_publish_frozen_holdouts_is_disjoint_immutable_and_audited(
    tmp_path: Path,
) -> None:
    challenge_sampling, challenge_labels = _challenge_inputs(per_category=1)
    natural_sampling = _natural_sampling_frame(prefix="natural")
    sampling = pl.concat(
        [challenge_sampling, natural_sampling], how="vertical"
    ).with_columns(
        pl.int_range(1, pl.len() + 1, dtype=pl.UInt32).alias("sampling_rank")
    )
    config = _config(challenge_per_category=1, natural_sample_size=4)
    challenge = build_balanced_challenge_holdout(
        sampling,
        challenge_labels,
        config,
    )
    selection = select_natural_stream_candidates(
        sampling,
        config,
        additionally_excluded_sampling_unit_ids=set(
            challenge["sampling_unit_id"].to_list()
        ),
    )
    natural_labels = _labels_for_selection(sampling, selection)
    natural = freeze_natural_stream_holdout(
        sampling,
        selection,
        natural_labels,
        config,
    )

    validate_evaluation_holdouts_disjoint(challenge, natural)
    publication = publish_frozen_evaluation_holdouts(
        challenge,
        natural,
        tmp_path / "frozen",
        run_id="frozen-holdout-test",
    )

    assert publication.balanced_challenge_path.name == BALANCED_CHALLENGE_HOLDOUT_FILE
    assert publication.natural_stream_path.name == NATURAL_STREAM_HOLDOUT_FILE
    assert publication.report_json_path.name == FROZEN_EVALUATION_HOLDOUT_REPORT_FILE
    assert (
        publication.report_markdown_path.name
        == FROZEN_EVALUATION_HOLDOUT_REPORT_MARKDOWN_FILE
    )
    report = json.loads(publication.report_json_path.read_text(encoding="utf-8"))
    assert report["balanced_challenge_rows"] == 7
    assert report["natural_stream_rows"] == 4
    assert report["natural_weight_sum"] == pytest.approx(10.0)
    assert report["artifacts"]["balanced_challenge"]["sha256"].startswith("sha256:")
    assert report["artifacts"]["natural_stream"]["sha256"].startswith("sha256:")
    with pytest.raises(FileExistsError):
        publish_frozen_evaluation_holdouts(
            challenge,
            natural,
            publication.output_dir,
        )


def test_disjoint_validator_rejects_exact_item_overlap() -> None:
    challenge_sampling, challenge_labels = _challenge_inputs(per_category=1)
    config = _config(challenge_per_category=1, natural_sample_size=1)
    challenge = build_balanced_challenge_holdout(
        challenge_sampling,
        challenge_labels,
        config,
    )
    selected_id = str(challenge["sampling_unit_id"][0])
    selection = select_natural_stream_candidates(
        challenge_sampling,
        config,
        additionally_excluded_sampling_unit_ids=(
            set(challenge["sampling_unit_id"].to_list()) - {selected_id}
        ),
    )
    labels = challenge_labels.filter(
        pl.col("flickr_photo_id") == selection["flickr_photo_id"][0]
    )
    natural = freeze_natural_stream_holdout(
        challenge_sampling,
        selection,
        labels,
        config,
    )

    with pytest.raises(ValueError, match="holdouts overlap"):
        validate_evaluation_holdouts_disjoint(challenge, natural)


def _config(
    *,
    challenge_per_category: int = 1,
    natural_sample_size: int = 4,
    minimum_per_natural_stratum: int = 1,
) -> FrozenHoldoutConfig:
    return FrozenHoldoutConfig(
        holdout_version="papilio-demoleus-evaluation-v1",
        target_accepted_taxon_key=TARGET,
        challenge_per_category=challenge_per_category,
        natural_sample_size=natural_sample_size,
        random_seed=73,
        minimum_per_natural_stratum=minimum_per_natural_stratum,
        natural_stratification_fields=("initial_visual_domain",),
    )


def _natural_sampling_frame(prefix: str = "stream") -> pl.DataFrame:
    rows = [
        _sampling_row(
            f"{prefix}-{index:02d}",
            rank=index + 1,
            visual_domain="artwork" if index == 9 else "live_field",
        )
        for index in range(10)
    ]
    return pl.DataFrame(rows, schema=EVALUATION_SAMPLING_FRAME_SCHEMA)


def _challenge_inputs(per_category: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    sampling_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    rank = 1
    for category in BALANCED_CHALLENGE_CATEGORIES:
        for index in range(per_category):
            photo_id = f"challenge-{category}-{index}"
            sampling = _sampling_row(
                photo_id,
                rank=rank,
                visual_domain=("artwork" if category == "artifacts" else "live_field"),
            )
            sampling_rows.append(sampling)
            label_rows.append(_label_row(sampling, category=category))
            rank += 1
    return (
        pl.DataFrame(sampling_rows, schema=EVALUATION_SAMPLING_FRAME_SCHEMA),
        pl.DataFrame(label_rows, schema=REVIEWED_LABEL_SCHEMA),
    )


def _labels_for_selection(
    sampling: pl.DataFrame,
    selection: pl.DataFrame,
) -> pl.DataFrame:
    selected = sampling.join(
        selection.select("sampling_unit_id"),
        on="sampling_unit_id",
        how="inner",
        validate="1:1",
    ).sort("sampling_unit_id")
    rows = [
        _label_row(
            row,
            category=(
                "verified_target" if index % 2 == 0 else "moths_and_other_insects"
            ),
        )
        for index, row in enumerate(selected.iter_rows(named=True))
    ]
    return pl.DataFrame(rows, schema=REVIEWED_LABEL_SCHEMA)


def _sampling_row(
    photo_id: str,
    *,
    rank: int,
    visual_domain: str,
) -> dict[str, object]:
    return {
        "schema_version": EVALUATION_SAMPLING_FRAME_SCHEMA_VERSION,
        "sampling_unit_id": f"sampling-unit:{photo_id}",
        "sampling_hash": f"sha256:{rank:064x}",
        "sampling_rank": rank,
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": f"sha256:record:{photo_id}",
        "photo_page_url": f"https://example.test/photos/{photo_id}",
        "image_url": f"https://example.test/images/{photo_id}.jpg",
        "owner_id": f"owner:{photo_id}",
        "owner_name": f"Owner {photo_id}",
        "source_owner_group_id": f"owner-group:{photo_id}",
        "year": 2025,
        "year_source": "date_taken",
        "year_stratum": "year:2025",
        "geo_cluster_id": "no_geo",
        "no_geo": True,
        "geo_stratum": "no_geo",
        "primary_query_tier": "T1",
        "query_tiers": ["T1"],
        "primary_query_term": "Papilio demoleus",
        "primary_query_field": "tags",
        "query_terms": ["Papilio demoleus"],
        "query_fields": ["tags"],
        "query_definition_ids": [f"query:{photo_id}"],
        "query_hit_count": 1,
        "query_provenance": [
            {
                "query_definition_id": f"query:{photo_id}",
                "query_tier": "T1",
                "query_term": "Papilio demoleus",
                "query_field": "tags",
                "query_priority": 10,
            }
        ],
        "metadata_target_text_evidence": False,
        "metadata_image_category": "unknown",
        "metadata_life_stage": "unknown",
        "initial_score_status": "not_scored",
        "initial_target_score_id": None,
        "initial_scoring_unit_id": None,
        "initial_route": "not_scored",
        "yoloe_route": "not_run",
        "yoloe_routes": [],
        "subject_area_ratio": None,
        "subject_area_band": "not_measured",
        "initial_visual_domain": visual_domain,
        "visual_domain_source": "fixture",
        "initial_reference_score": None,
        "initial_reference_score_band": "not_scored",
        "initial_reference_score_tail": "not_scored",
        "initial_competitor_margin": None,
        "initial_competitor_margin_band": "not_scored",
        "initial_competitor_margin_tail": "not_scored",
        "best_competitor_accepted_taxon_key": None,
        "best_competitor_scientific_name": None,
        "current_false_positive_genus": None,
        "false_positive_genus_stratum": "not_scored",
        "visual_input_disagreement": None,
        "visual_input_disagreement_band": "not_scored",
        "text_image_reference_disagreement": "reference_evidence_unavailable",
    }


def _label_row(
    sampling: dict[str, object],
    *,
    category: str,
) -> dict[str, object]:
    photo_id = str(sampling["flickr_photo_id"])
    row: dict[str, object] = {
        "schema_version": REVIEWED_LABEL_SCHEMA_VERSION,
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "detection_id": f"detection:{photo_id}",
        "crop_hash": f"sha256:crop:{photo_id}",
        "label_level": "species",
        "is_butterfly": True,
        "accepted_taxon_key": TARGET,
        "scientific_name": "Papilio demoleus",
        "family_key": "gbif:9417",
        "family": "Papilionidae",
        "genus_key": "gbif:papilio",
        "genus": "Papilio",
        "label_source": "manual_review",
        "reviewer_id": "reviewer-a",
        "reviewed_at": "2026-07-14T00:00:00Z",
        "review_confidence": "high",
        "review_notes": "synthetic reviewed evaluation item",
        "target_present": True,
        "label_certainty": "high",
        "life_stage": "adult",
        "visual_domain": "live_field",
        "view": "dorsal",
        "route": "adult_field",
        "geo_cluster_id": sampling["geo_cluster_id"],
        "source_query_tier": sampling["primary_query_tier"],
        "source_query_term": sampling["primary_query_term"],
        "duplicate_group_id": f"duplicate:{photo_id}",
        "observer_owner_group_id": sampling["source_owner_group_id"],
        "dataset_split": "final_test",
        "second_review_status": "completed",
        "ambiguity_reason": "",
        "unsuitable_for_species_identification": False,
    }
    if category == "other_papilio":
        row.update(
            target_present=False,
            accepted_taxon_key="gbif:other-papilio",
            scientific_name="Papilio polytes",
        )
    elif category == "other_papilionidae":
        row.update(
            target_present=False,
            accepted_taxon_key="gbif:graphium",
            scientific_name="Graphium agamemnon",
            genus_key="gbif:graphium-genus",
            genus="Graphium",
        )
    elif category == "moths_and_other_insects":
        row.update(
            label_level="negative",
            target_present=False,
            is_butterfly=False,
            accepted_taxon_key="gbif:moth",
            scientific_name="Noctua pronuba",
            family_key="gbif:erebidae",
            family="Erebidae",
            genus_key="gbif:noctua",
            genus="Noctua",
            route=None,
        )
    elif category == "artifacts":
        row.update(
            label_level="negative",
            target_present=False,
            is_butterfly=False,
            accepted_taxon_key=None,
            scientific_name=None,
            family_key=None,
            family=None,
            genus_key=None,
            genus=None,
            visual_domain="artwork",
            route=None,
            ambiguity_reason="artwork is not a biological observation",
            unsuitable_for_species_identification=True,
        )
    elif category == "pinned_specimens":
        row.update(
            target_present=False,
            accepted_taxon_key="gbif:pinned-other",
            scientific_name="Papilio polytes",
            visual_domain="pinned_specimen",
            life_stage="unknown",
            route="pinned_specimen",
        )
    elif category == "caterpillars":
        row.update(life_stage="larva", route="larval")
    elif category != "verified_target":
        raise ValueError(f"unsupported fixture category: {category}")
    return row
