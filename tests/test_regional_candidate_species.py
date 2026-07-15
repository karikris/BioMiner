from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from biominer.candidates.regional_union import (
    REGIONAL_CANDIDATE_SPECIES_FILE,
    REGIONAL_CANDIDATE_SPECIES_SCHEMA_VERSION,
    RegionalCandidateConfig,
    build_regional_candidate_species,
    regional_candidate_species_schema,
    write_regional_candidate_species,
)


TARGET = "gbif:1"
LOCAL_CONGENER = "gbif:2"
COUNTRY_CONGENER = "gbif:3"
LOCAL_FAMILY = "gbif:4"
GLOBAL_FAMILY = "gbif:5"
OTHER_FAMILY = "gbif:6"
REGISTRY_VERSION = "registry-test-v1"


def _taxa() -> pl.DataFrame:
    return pl.DataFrame(
        [
            _taxon(TARGET, "Papilio demoleus", "Papilionidae", "Papilio"),
            _taxon(LOCAL_CONGENER, "Papilio polytes", "Papilionidae", "Papilio"),
            _taxon(COUNTRY_CONGENER, "Papilio machaon", "Papilionidae", "Papilio"),
            _taxon(LOCAL_FAMILY, "Graphium agamemnon", "Papilionidae", "Graphium"),
            _taxon(GLOBAL_FAMILY, "Graphium sarpedon", "Papilionidae", "Graphium"),
            _taxon(OTHER_FAMILY, "Danaus plexippus", "Nymphalidae", "Danaus"),
        ]
    )


def _taxon(key: str, name: str, family: str, genus: str) -> dict[str, object]:
    return {
        "accepted_taxon_key": key,
        "scientific_name": name,
        "rank": "SPECIES",
        "taxonomic_status": "ACCEPTED",
        "family": family,
        "genus": genus,
        "in_scope": True,
    }


def _clusters(*cluster_ids: str) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "geo_cluster_id": cluster_id,
                "target_accepted_taxon_key": TARGET,
                "candidate_distribution_only": True,
                "cluster_configuration_hash": "sha256:" + "a" * 64,
            }
            for cluster_id in cluster_ids
        ]
    )


def _occurrence(
    cluster_id: str,
    taxon_key: str,
    overlap_type: str,
    *,
    count: int = 3,
    confidence: float | None = 0.8,
    source: str = "GBIF",
) -> dict[str, object]:
    taxon = next(row for row in _taxa().to_dicts() if row["accepted_taxon_key"] == taxon_key)
    return {
        "schema_version": "regional-taxon-occurrence-v1.0.0",
        "regional_scope_id": cluster_id,
        "regional_scope_type": "geo_cluster",
        "accepted_taxon_key": taxon_key,
        "scientific_name": taxon["scientific_name"],
        "family": taxon["family"],
        "subfamily": None,
        "tribe": None,
        "genus": taxon["genus"],
        "occurrence_count": count,
        "independent_dataset_count": 1,
        "earliest_occurrence_date": None,
        "latest_occurrence_date": None,
        "coordinate_confidence": confidence,
        "overlap_type": overlap_type,
        "source": source,
        "source_dataset_keys": [f"{source.casefold()}-dataset"],
        "evidence_version": "regional-evidence-v1",
        "registry_version": REGISTRY_VERSION,
    }


def _relationships() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "subject_accepted_taxon_key": TARGET,
                "object_scope_type": "species",
                "object_scope_id": OTHER_FAMILY,
                "relationship_type": "known_mimic",
                "evidence_version": "relationships-v1",
                "review_status": "reviewed",
                "enabled": True,
            },
            {
                "subject_accepted_taxon_key": TARGET,
                "object_scope_type": "genus",
                "object_scope_id": "Graphium",
                "relationship_type": "historical_false_positive_genus",
                "evidence_version": "relationships-v1",
                "review_status": "reviewed",
                "enabled": True,
            },
        ]
    )


def test_builds_union_with_all_reasons_flags_and_soft_geography() -> None:
    occurrences = pl.DataFrame(
        [
            _occurrence("cluster-a", TARGET, "exact", count=4, confidence=0.9),
            _occurrence("cluster-a", LOCAL_CONGENER, "buffer", count=3, confidence=0.8),
            _occurrence("cluster-a", LOCAL_FAMILY, "exact", count=2, confidence=0.7),
            _occurrence("cluster-a", COUNTRY_CONGENER, "country", count=5, confidence=0.6),
            _occurrence("cluster-a", GLOBAL_FAMILY, "country", count=1, confidence=0.5),
            _occurrence("cluster-a", OTHER_FAMILY, "exact", count=20, confidence=1.0),
        ]
    )

    result = build_regional_candidate_species(
        target_accepted_taxon_key=TARGET,
        geo_clusters=_clusters("cluster-a"),
        regional_occurrence=occurrences,
        taxa=_taxa(),
        registry_version=REGISTRY_VERSION,
        competitor_relationships=_relationships(),
        historical_false_positive_taxon_keys=[LOCAL_CONGENER],
        historical_false_positive_version="false-positives-v1",
        visually_nearest_taxon_keys=[COUNTRY_CONGENER],
        visual_neighbour_version="visual-neighbours-v1",
        config=RegionalCandidateConfig(minimum_local_same_family_candidates=4),
    )

    assert result.schema == regional_candidate_species_schema()
    by_key = {row["candidate_accepted_taxon_key"]: row for row in result.to_dicts()}
    assert set(by_key) == {
        TARGET,
        LOCAL_CONGENER,
        COUNTRY_CONGENER,
        LOCAL_FAMILY,
        GLOBAL_FAMILY,
        OTHER_FAMILY,
    }
    assert by_key[TARGET]["target_candidate"] is True
    assert by_key[TARGET]["candidate_reason"] == ["target"]
    assert by_key[TARGET]["candidate_priority"] == 0
    assert by_key[LOCAL_CONGENER]["candidate_reason"] == [
        "historical_false_positive",
        "same_genus_range_overlap",
        "regional_same_family",
    ]
    assert by_key[LOCAL_CONGENER]["historical_false_positive"] is True
    assert by_key[COUNTRY_CONGENER]["candidate_reason"] == [
        "visually_nearest",
        "same_genus_range_overlap",
        "country_fallback",
    ]
    assert by_key[COUNTRY_CONGENER]["visually_nearest"] is True
    assert by_key[GLOBAL_FAMILY]["candidate_reason"] == [
        "historical_false_positive",
        "country_fallback",
    ]
    assert by_key[OTHER_FAMILY]["candidate_reason"] == ["known_mimic"]
    assert by_key[OTHER_FAMILY]["known_mimic"] is True
    assert by_key[OTHER_FAMILY]["same_family"] is False
    assert by_key[LOCAL_CONGENER]["geographic_evidence_score"] == pytest.approx(0.64)
    assert by_key[LOCAL_CONGENER]["occurrence_support"] == 3
    assert by_key[OTHER_FAMILY]["geographic_evidence_score"] == pytest.approx(1.0)
    assert len({row["candidate_set_id"] for row in by_key.values()}) == 1
    assert len({row["candidate_set_fingerprint"] for row in by_key.values()}) == 1
    assert all(
        row["schema_version"] == REGIONAL_CANDIDATE_SPECIES_SCHEMA_VERSION
        for row in by_key.values()
    )
    assert by_key[TARGET]["source_versions"] == [
        "candidate-config:min-local=4;registry-global-fallback=true",
        "candidate-policy:regional-candidate-union-v1.1.0",
        "false-positives:false-positives-v1",
        "flickr-clusters:sha256:" + "a" * 64,
        "geographic-score:overlap-weighted-coordinate-evidence-v1.0.0",
        "regional-occurrence:regional-evidence-v1",
        "registry:registry-test-v1",
        "relationships:relationships-v1",
        "visual-neighbours:visual-neighbours-v1",
    ]


def test_dense_local_set_omits_broad_same_family_but_keeps_broad_congener() -> None:
    occurrences = pl.DataFrame(
        [
            _occurrence("cluster-a", LOCAL_CONGENER, "exact"),
            _occurrence("cluster-a", LOCAL_FAMILY, "buffer"),
            _occurrence("cluster-a", COUNTRY_CONGENER, "country"),
            _occurrence("cluster-a", GLOBAL_FAMILY, "country"),
        ]
    )

    result = build_regional_candidate_species(
        target_accepted_taxon_key=TARGET,
        geo_clusters=_clusters("cluster-a"),
        regional_occurrence=occurrences,
        taxa=_taxa(),
        registry_version=REGISTRY_VERSION,
        config=RegionalCandidateConfig(minimum_local_same_family_candidates=2),
    )
    keys = set(result["candidate_accepted_taxon_key"].to_list())

    assert COUNTRY_CONGENER in keys
    assert GLOBAL_FAMILY not in keys
    country = result.filter(
        pl.col("candidate_accepted_taxon_key") == COUNTRY_CONGENER
    ).to_dicts()[0]
    assert country["candidate_reason"] == ["same_genus_range_overlap"]


def test_no_geo_expands_to_global_same_family_without_deleting_target() -> None:
    occurrences = pl.DataFrame(
        [
            _occurrence("no_geo", LOCAL_CONGENER, "global", count=8),
            _occurrence("no_geo", OTHER_FAMILY, "global", count=99),
        ]
    )

    result = build_regional_candidate_species(
        target_accepted_taxon_key=TARGET,
        geo_clusters=_clusters("no_geo"),
        regional_occurrence=occurrences,
        taxa=_taxa(),
        registry_version=REGISTRY_VERSION,
    )
    by_key = {row["candidate_accepted_taxon_key"]: row for row in result.to_dicts()}

    assert set(by_key) == {
        TARGET,
        LOCAL_CONGENER,
        COUNTRY_CONGENER,
        LOCAL_FAMILY,
        GLOBAL_FAMILY,
    }
    assert OTHER_FAMILY not in by_key
    assert by_key[TARGET]["target_candidate"] is True
    assert by_key[TARGET]["candidate_reason"] == ["target", "global_registry_fallback"]
    assert by_key[LOCAL_CONGENER]["candidate_reason"] == [
        "same_genus_range_overlap",
        "global_no_geo_fallback",
        "global_registry_fallback",
    ]
    assert by_key[LOCAL_CONGENER]["occurrence_support"] == 8
    assert by_key[GLOBAL_FAMILY]["geographic_evidence_score"] is None


def test_unassigned_geo_uses_global_fallback_without_claiming_missing_coordinates(
) -> None:
    result = build_regional_candidate_species(
        target_accepted_taxon_key=TARGET,
        geo_clusters=_clusters("unassigned_geo"),
        regional_occurrence=pl.DataFrame(
            [
                _occurrence(
                    "unassigned_geo",
                    LOCAL_CONGENER,
                    "global",
                    count=8,
                )
            ]
        ),
        taxa=_taxa(),
        registry_version=REGISTRY_VERSION,
    )
    by_key = {row["candidate_accepted_taxon_key"]: row for row in result.to_dicts()}

    assert "global_unassigned_geo_fallback" in by_key[LOCAL_CONGENER][
        "candidate_reason"
    ]
    assert "global_no_geo_fallback" not in by_key[LOCAL_CONGENER][
        "candidate_reason"
    ]


def test_target_is_only_candidate_when_no_evidence_or_competitors_exist() -> None:
    result = build_regional_candidate_species(
        target_accepted_taxon_key=TARGET,
        geo_clusters=_clusters("cluster-empty"),
        regional_occurrence=pl.DataFrame(),
        taxa=_taxa(),
        registry_version=REGISTRY_VERSION,
    )

    assert result.height == 1
    row = result.to_dicts()[0]
    assert row["candidate_accepted_taxon_key"] == TARGET
    assert row["target_candidate"] is True
    assert row["candidate_reason"] == ["target"]


def test_rejects_unaccepted_competitors_and_unreviewed_enabled_relationships() -> None:
    with pytest.raises(ValueError, match="not accepted species"):
        build_regional_candidate_species(
            target_accepted_taxon_key=TARGET,
            geo_clusters=_clusters("cluster-a"),
            regional_occurrence=pl.DataFrame(),
            taxa=_taxa(),
            registry_version=REGISTRY_VERSION,
            historical_false_positive_taxon_keys=["gbif:missing"],
        )

    with pytest.raises(ValueError, match="historical_false_positive_version is required"):
        build_regional_candidate_species(
            target_accepted_taxon_key=TARGET,
            geo_clusters=_clusters("cluster-a"),
            regional_occurrence=pl.DataFrame(),
            taxa=_taxa(),
            registry_version=REGISTRY_VERSION,
            historical_false_positive_taxon_keys=[LOCAL_CONGENER],
        )

    unreviewed = _relationships().with_columns(pl.lit("pending").alias("review_status"))
    with pytest.raises(ValueError, match="enabled competitor relationship must be reviewed"):
        build_regional_candidate_species(
            target_accepted_taxon_key=TARGET,
            geo_clusters=_clusters("cluster-a"),
            regional_occurrence=pl.DataFrame(),
            taxa=_taxa(),
            registry_version=REGISTRY_VERSION,
            competitor_relationships=unreviewed,
        )


def test_is_deterministic_and_writes_exact_schema(tmp_path: Path) -> None:
    occurrences = pl.DataFrame(
        [
            _occurrence("cluster-b", LOCAL_FAMILY, "exact"),
            _occurrence("cluster-a", LOCAL_CONGENER, "exact"),
        ]
    )
    first = build_regional_candidate_species(
        target_accepted_taxon_key=TARGET,
        geo_clusters=_clusters("cluster-b", "cluster-a"),
        regional_occurrence=occurrences,
        taxa=_taxa(),
        registry_version=REGISTRY_VERSION,
    )
    second = build_regional_candidate_species(
        target_accepted_taxon_key=TARGET,
        geo_clusters=_clusters("cluster-a", "cluster-b").reverse(),
        regional_occurrence=occurrences.reverse(),
        taxa=_taxa().reverse(),
        registry_version=REGISTRY_VERSION,
    )

    assert first.equals(second)
    output = write_regional_candidate_species(first, tmp_path)
    assert output == tmp_path / REGIONAL_CANDIDATE_SPECIES_FILE
    restored = pl.read_parquet(output)
    assert restored.equals(first)
    assert restored.schema == regional_candidate_species_schema()
    for (_set_id,), group in restored.group_by("candidate_set_id"):
        assert group["candidate_priority"].to_list() == list(range(group.height))
        assert group["target_candidate"].sum() == 1
