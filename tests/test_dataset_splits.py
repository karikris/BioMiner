from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from biominer.evaluation.splits import (
    DATASET_SPLIT_MANIFEST_FILE,
    DATASET_SPLIT_MANIFEST_SCHEMA_VERSION,
    DATASET_SPLITS,
    DatasetSplitConfig,
    DatasetSplitItem,
    build_dataset_split_manifest,
    dataset_split_manifest_schema,
    load_dataset_split_manifest,
    validate_dataset_split_manifest,
    write_dataset_split_manifest,
)
from biominer.ml.training_features import build_few_shot_training_features
from test_few_shot_training_features import _example, _sha


TARGET = "gbif:target"
COMPETITOR = "gbif:competitor"


def test_build_is_deterministic_balanced_and_exactly_versioned() -> None:
    items = _items(classes=(TARGET, COMPETITOR), components_per_class=12)
    config = DatasetSplitConfig(split_version="papilio-pilot-v1", random_seed=73)

    first = build_dataset_split_manifest(items, config)
    second = build_dataset_split_manifest(tuple(reversed(items)), config)

    assert_frame_equal(first.manifest, second.manifest)
    assert first.manifest.schema == dataset_split_manifest_schema()
    assert first.manifest["schema_version"].unique().to_list() == [
        DATASET_SPLIT_MANIFEST_SCHEMA_VERSION
    ]
    assert first.manifest["split_fingerprint"].n_unique() == 1
    assert first.split_fingerprint.startswith("sha256:")
    assert first.configuration_fingerprint == config.fingerprint
    assert first.component_count == len(items)
    assert first.manifest["item_id"].to_list() == sorted(
        first.manifest["item_id"].to_list()
    )
    assert set(first.manifest["dataset_split"]) == set(DATASET_SPLITS)

    counts_by_class_and_split: dict[str, dict[str, int]] = defaultdict(dict)
    for row in (
        first.manifest.group_by("stratification_label", "dataset_split")
        .agg(pl.len().alias("count"))
        .iter_rows(named=True)
    ):
        counts_by_class_and_split[str(row["stratification_label"])][
            str(row["dataset_split"])
        ] = int(row["count"])
    for class_label in (TARGET, COMPETITOR):
        assert set(counts_by_class_and_split[class_label]) == set(DATASET_SPLITS)
        assert max(counts_by_class_and_split[class_label].values()) <= 7


def test_transitive_identity_links_form_one_atomic_component() -> None:
    items = list(_items(classes=(TARGET,), components_per_class=17))
    items[0] = replace(items[0], source_observation_id="observation:chain")
    items[1] = replace(
        items[1],
        source_observation_id="observation:chain",
        duplicate_group_id="duplicate:chain",
    )
    items[2] = replace(
        items[2],
        duplicate_group_id="duplicate:chain",
        exact_hash_group_id="exact:chain",
    )
    items[3] = replace(
        items[3],
        exact_hash_group_id="exact:chain",
        perceptual_duplicate_group_id="perceptual:chain",
    )
    items[4] = replace(
        items[4],
        perceptual_duplicate_group_id="perceptual:chain",
        observer_id="person:chain",
    )
    items[5] = replace(
        items[5],
        photographer_id="person:chain",
        flickr_owner_id="flickr-owner:chain",
    )
    items[6] = replace(
        items[6],
        flickr_owner_id="flickr-owner:chain",
        burst_group_id="burst:chain",
    )
    items[7] = replace(
        items[7],
        burst_group_id="burst:chain",
        provider_mirror_group_id="mirror:chain",
    )
    items[8] = replace(
        items[8],
        provider_mirror_group_id="mirror:chain",
        geo_cluster_id="geo:chain",
    )
    items[9] = replace(items[9], geo_cluster_id="geo:chain")

    result = build_dataset_split_manifest(
        tuple(items),
        DatasetSplitConfig(split_version="transitive-v1", random_seed=11),
    )
    chain = result.manifest.filter(
        pl.col("item_id").is_in([item.item_id for item in items[:10]])
    )

    assert chain["leakage_component_id"].n_unique() == 1
    assert chain["dataset_split"].n_unique() == 1
    assert chain["leakage_component_size"].unique().to_list() == [10]
    assert result.component_count == 8


def test_no_geo_is_missing_not_a_shared_leakage_component() -> None:
    items = tuple(
        replace(item, geo_cluster_id="no_geo")
        for item in _items(classes=(TARGET,), components_per_class=8)
    )

    result = build_dataset_split_manifest(
        items,
        DatasetSplitConfig(split_version="no-geo-v1", random_seed=5),
    )

    assert result.component_count == len(items)
    assert result.manifest["leakage_component_id"].n_unique() == len(items)
    assert set(result.manifest["dataset_split"]) == set(DATASET_SPLITS)


def test_mixed_class_components_remain_atomic_and_preserve_class_coverage() -> None:
    items = list(_items(classes=(TARGET, COMPETITOR), components_per_class=8))
    for index in range(4):
        shared_group = f"mirror:mixed:{index}"
        items[index] = replace(
            items[index],
            provider_mirror_group_id=shared_group,
        )
        items[8 + index] = replace(
            items[8 + index],
            provider_mirror_group_id=shared_group,
        )

    result = build_dataset_split_manifest(
        tuple(items),
        DatasetSplitConfig(split_version="mixed-class-v1", random_seed=29),
    )

    assert result.component_count == 12
    for split_name in DATASET_SPLITS:
        labels = set(
            result.manifest.filter(pl.col("dataset_split") == split_name)[
                "stratification_label"
            ]
        )
        assert labels == {TARGET, COMPETITOR}


def test_person_identifiers_are_shared_across_roles_but_scoped_by_source() -> None:
    items = list(_items(classes=(TARGET,), components_per_class=8))
    items[0] = replace(items[0], observer_id="person:7")
    items[1] = replace(items[1], photographer_id="person:7")
    items[2] = replace(
        items[2],
        source="inaturalist",
        observer_id="person:7",
        flickr_owner_id=None,
    )

    result = build_dataset_split_manifest(
        tuple(items),
        DatasetSplitConfig(
            split_version="person-scope-v1",
            random_seed=19,
            require_class_coverage=False,
        ),
    )
    by_id = {str(row["item_id"]): row for row in result.manifest.iter_rows(named=True)}

    assert (
        by_id[items[0].item_id]["leakage_component_id"]
        == by_id[items[1].item_id]["leakage_component_id"]
    )
    assert (
        by_id[items[2].item_id]["leakage_component_id"]
        != by_id[items[0].item_id]["leakage_component_id"]
    )


def test_rejects_missing_flickr_owner_and_insufficient_class_components() -> None:
    item = _items(classes=(TARGET,), components_per_class=1)[0]
    with pytest.raises(ValueError, match="flickr_owner_id"):
        replace(item, flickr_owner_id=None)

    sparse = _items(classes=(TARGET, COMPETITOR), components_per_class=3)
    with pytest.raises(ValueError, match="independent components.*four partitions"):
        build_dataset_split_manifest(
            sparse,
            DatasetSplitConfig(split_version="sparse-v1"),
        )


def test_seed_changes_assignment_identity_but_not_leakage_components() -> None:
    items = _items(classes=(TARGET, COMPETITOR), components_per_class=12)
    first = build_dataset_split_manifest(
        items,
        DatasetSplitConfig(split_version="seed-v1", random_seed=1),
    )
    second = build_dataset_split_manifest(
        items,
        DatasetSplitConfig(split_version="seed-v1", random_seed=2),
    )

    first_components = first.manifest.select("item_id", "leakage_component_id")
    second_components = second.manifest.select("item_id", "leakage_component_id")
    assert_frame_equal(first_components, second_components)
    assert first.configuration_fingerprint != second.configuration_fingerprint
    assert first.split_fingerprint != second.split_fingerprint


def test_validator_rejects_cross_split_component_and_fingerprint_tampering() -> None:
    items = list(_items(classes=(TARGET,), components_per_class=10))
    items[1] = replace(
        items[1],
        duplicate_group_id=items[0].duplicate_group_id,
    )
    result = build_dataset_split_manifest(
        tuple(items),
        DatasetSplitConfig(split_version="tamper-v1", random_seed=31),
    )
    shared_component = str(result.manifest["leakage_component_id"][0])
    shared = result.manifest.filter(pl.col("leakage_component_id") == shared_component)
    assert shared.height == 2
    original_split = str(shared["dataset_split"][0])
    different_split = next(value for value in DATASET_SPLITS if value != original_split)
    tampered_split = result.manifest.with_columns(
        pl.when(pl.col("item_id") == shared["item_id"][0])
        .then(pl.lit(different_split))
        .otherwise(pl.col("dataset_split"))
        .alias("dataset_split")
    )
    with pytest.raises(ValueError, match="leakage component.*crosses splits"):
        validate_dataset_split_manifest(tampered_split)

    tampered_fingerprint = result.manifest.with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(pl.lit(_sha("tampered-split")))
        .otherwise(pl.col("split_fingerprint"))
        .alias("split_fingerprint")
    )
    with pytest.raises(ValueError, match="split_fingerprint"):
        validate_dataset_split_manifest(tampered_fingerprint)


def test_parquet_publication_is_immutable_and_round_trips(tmp_path: Path) -> None:
    result = build_dataset_split_manifest(
        _items(classes=(TARGET, COMPETITOR), components_per_class=8),
        DatasetSplitConfig(split_version="publish-v1", random_seed=101),
    )

    path = write_dataset_split_manifest(result, tmp_path / "published")
    loaded = load_dataset_split_manifest(
        path,
        expected_split_fingerprint=result.split_fingerprint,
        expected_configuration_fingerprint=result.configuration_fingerprint,
    )

    assert path.name == DATASET_SPLIT_MANIFEST_FILE
    assert_frame_equal(loaded, result.manifest)
    with pytest.raises(FileExistsError):
        write_dataset_split_manifest(result, tmp_path / "published")


def test_training_feature_validation_rejects_real_geo_leakage_and_skips_no_geo() -> (
    None
):
    first = _example("geo-a", group_id="group-a")
    second = _example("geo-b", group_id="group-b")
    second = replace(
        second,
        provenance=replace(
            second.provenance,
            dataset_split="model_selection",
            source_owner_id="owner:geo-b",
        ),
    )
    with pytest.raises(ValueError, match="geo_cluster_id.*crosses dataset splits"):
        build_few_shot_training_features((first, second))

    no_geo_first = _example("no-geo-a", group_id="no-geo-group-a", missing_geo=True)
    no_geo_second = _example("no-geo-b", group_id="no-geo-group-b", missing_geo=True)
    no_geo_second = replace(
        no_geo_second,
        provenance=replace(
            no_geo_second.provenance,
            dataset_split="model_selection",
            source_owner_id="owner:no-geo-b",
        ),
    )

    frame = build_few_shot_training_features((no_geo_first, no_geo_second))
    assert set(frame["dataset_split"]) == {"support_train", "model_selection"}


def _items(
    *,
    classes: tuple[str, ...],
    components_per_class: int,
) -> tuple[DatasetSplitItem, ...]:
    result: list[DatasetSplitItem] = []
    for class_index, class_label in enumerate(classes):
        for index in range(components_per_class):
            suffix = f"{class_index:02d}:{index:03d}"
            result.append(
                DatasetSplitItem(
                    item_type="reviewed_visual_input",
                    item_id=f"item:{suffix}",
                    source="flickr",
                    route="adult_field",
                    stratification_label=class_label,
                    accepted_class_taxon_key=class_label,
                    source_artifact_fingerprint=_sha("reviewed-labels-v2"),
                    source_observation_id=f"observation:{suffix}",
                    source_owner_id=f"owner:{suffix}",
                    observer_id=f"observer:{suffix}",
                    photographer_id=f"photographer:{suffix}",
                    flickr_owner_id=f"flickr-owner:{suffix}",
                    duplicate_group_id=f"duplicate:{suffix}",
                    exact_hash_group_id=f"exact:{suffix}",
                    perceptual_duplicate_group_id=f"perceptual:{suffix}",
                    burst_group_id=f"burst:{suffix}",
                    provider_mirror_group_id=f"mirror:{suffix}",
                    geo_cluster_id=f"geo:{suffix}",
                )
            )
    return tuple(result)
