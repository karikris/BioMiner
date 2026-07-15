from __future__ import annotations

from datetime import UTC, datetime
import json

import polars as pl
import pytest

from biominer.references.regional_competitors import (
    REGIONAL_COMPETITOR_EVIDENCE_FILE,
    REGIONAL_COMPETITOR_MANIFEST_FILE,
    RegionalCompetitorBuildResult,
    SpeciesFacetCount,
    build_regional_competitor_evidence,
    regional_competitor_evidence_schema,
    write_regional_competitor_artifacts,
)


class FakeFacetSource:
    source = "GBIF"
    candidate_genus_taxon_key = "gbif:10"
    source_snapshot_version = "gbif-fixture"
    attempt_count = 0
    retry_count = 0
    rate_limit_count = 0

    def __init__(self, responses: dict[str, tuple[SpeciesFacetCount, ...]]) -> None:
        self.responses = responses
        self.requested: list[str] = []

    def country_species_counts(
        self,
        country_code: str,
    ) -> tuple[SpeciesFacetCount, ...]:
        self.requested.append(country_code)
        self.attempt_count += 1
        return self.responses[country_code]


def _taxa() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:1",
                "scientific_name": "Papilio target",
                "rank": "SPECIES",
                "taxonomic_status": "ACCEPTED",
                "family": "Papilionidae",
                "genus": "Papilio",
                "genus_key": "gbif:10",
            },
            {
                "accepted_taxon_key": "gbif:2",
                "scientific_name": "Papilio widespread",
                "rank": "SPECIES",
                "taxonomic_status": "ACCEPTED",
                "family": "Papilionidae",
                "genus": "Papilio",
                "genus_key": "gbif:10",
            },
            {
                "accepted_taxon_key": "gbif:3",
                "scientific_name": "Papilio local",
                "rank": "SPECIES",
                "taxonomic_status": "ACCEPTED",
                "family": "Papilionidae",
                "genus": "Papilio",
                "genus_key": "gbif:10",
            },
            {
                "accepted_taxon_key": "gbif:4",
                "scientific_name": "Papilio synonym",
                "rank": "SPECIES",
                "taxonomic_status": "SYNONYM",
                "family": "Papilionidae",
                "genus": "Papilio",
                "genus_key": "gbif:10",
            },
            {
                "accepted_taxon_key": "gbif:10",
                "scientific_name": "Papilio",
                "rank": "GENUS",
                "taxonomic_status": "ACCEPTED",
                "family": "Papilionidae",
                "genus": "Papilio",
                "genus_key": "gbif:10",
            },
            {
                "accepted_taxon_key": "gbif:20",
                "scientific_name": "Othergenus",
                "rank": "GENUS",
                "taxonomic_status": "ACCEPTED",
                "family": "Otheridae",
                "genus": "Othergenus",
                "genus_key": "gbif:20",
            },
        ]
    )


def _occurrences() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for country, count in (("IN", 5), ("TH", 3), ("AU", 1)):
        for index in range(count):
            rows.extend(
                [
                    {
                        "gbif_id": f"{country}-{index}",
                        "accepted_taxon_key": "gbif:1",
                        "country_code": country,
                        "range_inference_eligible": True,
                    },
                    {
                        "gbif_id": f"{country}-{index}",
                        "accepted_taxon_key": "gbif:1",
                        "country_code": country,
                        "range_inference_eligible": True,
                    },
                ]
            )
    rows.append(
        {
            "gbif_id": "excluded",
            "accepted_taxon_key": "gbif:1",
            "country_code": "IN",
            "range_inference_eligible": False,
        }
    )
    return pl.DataFrame(rows)


def _source() -> FakeFacetSource:
    return FakeFacetSource(
        {
            "IN": (
                SpeciesFacetCount("gbif:1", 100),
                SpeciesFacetCount("gbif:2", 20),
                SpeciesFacetCount("gbif:3", 40),
                SpeciesFacetCount("gbif:4", 50),
                SpeciesFacetCount("gbif:999", 60),
            ),
            "TH": (
                SpeciesFacetCount("gbif:1", 80),
                SpeciesFacetCount("gbif:2", 10),
            ),
        }
    )


def _build(tmp_path, source: FakeFacetSource) -> RegionalCompetitorBuildResult:
    return build_regional_competitor_evidence(
        target_occurrence_evidence=_occurrences(),
        taxa=_taxa(),
        target_accepted_taxon_key="gbif:1",
        candidate_genus_taxon_key="gbif:10",
        registry_version="registry-fixture",
        source_snapshot_version="gbif-fixture",
        source=source,  # type: ignore[arg-type]
        checkpoint_dir=tmp_path / "checkpoints",
        retrieved_at="2026-07-15T00:00:00Z",
        minimum_target_country_occurrences=3,
        maximum_candidates=10,
    )


def test_build_compiles_registry_validated_country_overlap_and_checkpoints(
    tmp_path,
) -> None:
    source = _source()
    result = _build(tmp_path, source)

    assert result.evidence.schema == regional_competitor_evidence_schema()
    assert source.requested == ["IN", "TH"]
    assert result.evidence.select(
        "candidate_accepted_taxon_key", "candidate_rank", "overlap_country_count"
    ).to_dicts() == [
        {
            "candidate_accepted_taxon_key": "gbif:2",
            "candidate_rank": 1,
            "overlap_country_count": 2,
        },
        {
            "candidate_accepted_taxon_key": "gbif:3",
            "candidate_rank": 2,
            "overlap_country_count": 1,
        },
    ]
    assert result.evidence["candidate_occurrence_count"].to_list() == [30, 40]
    assert result.manifest["target_country_count"] == 2
    assert result.manifest["unmatched_or_out_of_scope_facet_keys"] == [
        "gbif:4",
        "gbif:999",
    ]
    assert result.manifest["api"] == {
        "successful_query_count": 2,
        "attempt_count": 2,
        "retry_count": 0,
        "rate_limit_count": 0,
    }
    assert result.manifest["pid"] > 0
    assert result.manifest["elapsed_seconds"] >= 0
    assert result.manifest["checkpoint"] == {
        "resumed": False,
        "initial_completed_country_count": 0,
        "completed_country_count": 2,
        "state_file": str(tmp_path / "checkpoints" / "state.json"),
    }


def test_complete_checkpoint_resumes_without_refetching(tmp_path) -> None:
    first = _build(tmp_path, _source())
    resumed_source = FakeFacetSource({})
    second = _build(tmp_path, resumed_source)

    assert first.evidence.equals(second.evidence)
    assert second.resumed is True
    assert resumed_source.requested == []
    assert second.manifest["checkpoint"]["initial_completed_country_count"] == 2


def test_checkpoint_identity_mismatch_fails_closed(tmp_path) -> None:
    _build(tmp_path, _source())

    with pytest.raises(ValueError, match="identity mismatch"):
        build_regional_competitor_evidence(
            target_occurrence_evidence=_occurrences(),
            taxa=_taxa(),
            target_accepted_taxon_key="gbif:1",
            candidate_genus_taxon_key="gbif:10",
            registry_version="different-registry",
            source_snapshot_version="gbif-fixture",
            source=_source(),  # type: ignore[arg-type]
            checkpoint_dir=tmp_path / "checkpoints",
            retrieved_at="2026-07-15T00:00:00Z",
            minimum_target_country_occurrences=3,
        )


def test_write_artifacts_records_bytes_and_checksum(tmp_path) -> None:
    result = _build(tmp_path / "build", _source())
    artifacts = write_regional_competitor_artifacts(result, tmp_path / "output")

    assert artifacts["evidence"].name == REGIONAL_COMPETITOR_EVIDENCE_FILE
    assert artifacts["manifest"].name == REGIONAL_COMPETITOR_MANIFEST_FILE
    restored = pl.read_parquet(artifacts["evidence"])
    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
    assert restored.equals(result.evidence)
    assert manifest["files"]["regional_competitor_evidence"]["row_count"] == 2
    assert manifest["files"]["regional_competitor_evidence"]["byte_count"] > 0
    assert manifest["files"]["regional_competitor_evidence"]["sha256"].startswith(
        "sha256:"
    )


def test_rejects_naive_retrieval_timestamp(tmp_path) -> None:
    with pytest.raises(ValueError, match="UTC offset"):
        build_regional_competitor_evidence(
            target_occurrence_evidence=_occurrences(),
            taxa=_taxa(),
            target_accepted_taxon_key="gbif:1",
            candidate_genus_taxon_key="gbif:10",
            registry_version="registry-fixture",
            source_snapshot_version="gbif-fixture",
            source=_source(),  # type: ignore[arg-type]
            checkpoint_dir=tmp_path / "checkpoints",
            retrieved_at=datetime(2026, 7, 15),
            minimum_target_country_occurrences=3,
        )


def test_accepts_aware_retrieval_timestamp(tmp_path) -> None:
    result = build_regional_competitor_evidence(
        target_occurrence_evidence=_occurrences(),
        taxa=_taxa(),
        target_accepted_taxon_key="gbif:1",
        candidate_genus_taxon_key="gbif:10",
        registry_version="registry-fixture",
        source_snapshot_version="gbif-fixture",
        source=_source(),  # type: ignore[arg-type]
        checkpoint_dir=tmp_path / "checkpoints",
        retrieved_at=datetime(2026, 7, 15, tzinfo=UTC),
        minimum_target_country_occurrences=3,
    )

    assert result.evidence.schema["retrieved_at"] == pl.Datetime("us", "UTC")


def test_rejects_candidate_genus_outside_target_family(tmp_path) -> None:
    source = _source()
    source.candidate_genus_taxon_key = "gbif:20"

    with pytest.raises(ValueError, match="target species family"):
        build_regional_competitor_evidence(
            target_occurrence_evidence=_occurrences(),
            taxa=_taxa(),
            target_accepted_taxon_key="gbif:1",
            candidate_genus_taxon_key="gbif:20",
            registry_version="registry-fixture",
            source_snapshot_version="gbif-fixture",
            source=source,  # type: ignore[arg-type]
            checkpoint_dir=tmp_path / "checkpoints",
            retrieved_at="2026-07-15T00:00:00Z",
            minimum_target_country_occurrences=3,
        )
