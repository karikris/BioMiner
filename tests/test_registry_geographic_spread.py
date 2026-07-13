from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest

from biominer.geography import GeographicResolutions
from biominer.registry.geographic_spread import (
    GBIF_SEARCH_MAX_RECORDS,
    GEOGRAPHIC_SPREAD_SCHEMA_VERSION,
    TAXON_GEOGRAPHIC_SPREAD_FILE,
    BulkDownloadRequired,
    GBIFOccurrenceSearchSource,
    GBIFParquetOccurrenceSource,
    OccurrenceBatch,
    build_gbif_bulk_download_request,
    build_taxon_geographic_spread,
    geographic_spread_schema,
)


TARGET_KEY = "gbif:1938069"
SNAPSHOT_VERSION = "gbif-occurrence-search-2026-07-13"
RETRIEVED_AT = "2026-07-13T00:00:00Z"


class FakeHTTPGet:
    def __init__(self, payloads: dict[int, dict[str, object]]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, path: str, params: dict[str, object]) -> dict[str, object]:
        self.calls.append((path, dict(params)))
        return self.payloads[int(params["offset"])]


class FakeBatchSource:
    source = "GBIF"

    def __init__(
        self,
        batches: tuple[OccurrenceBatch, ...],
        *,
        fail_after_batches: int | None = None,
    ) -> None:
        self.batches = batches
        self.fail_after_batches = fail_after_batches
        self.source_query_hash = "sha256:" + ("1" * 64)
        self.source_snapshot_version = SNAPSHOT_VERSION
        self.start_cursors: list[int] = []

    def iter_batches(self, *, start_cursor: int = 0) -> Iterator[OccurrenceBatch]:
        self.start_cursors.append(start_cursor)
        yielded = 0
        for batch in self.batches:
            if batch.cursor < start_cursor:
                continue
            if self.fail_after_batches is not None and yielded >= self.fail_after_batches:
                raise RuntimeError("injected source interruption")
            yielded += 1
            yield batch


def test_search_source_pages_with_accepted_taxon_and_coordinate_filter() -> None:
    http = FakeHTTPGet(
        {
            0: {
                "count": 3,
                "offset": 0,
                "limit": 2,
                "endOfRecords": False,
                "results": [_occurrence("1"), _occurrence("2")],
            },
            2: {
                "count": 3,
                "offset": 2,
                "limit": 2,
                "endOfRecords": True,
                "results": [_occurrence("3")],
            },
        }
    )
    source = GBIFOccurrenceSearchSource(
        accepted_taxon_key=TARGET_KEY,
        source_snapshot_version=SNAPSHOT_VERSION,
        http_get=http,
        page_size=2,
    )

    batches = tuple(source.iter_batches())

    assert [batch.cursor for batch in batches] == [0, 2]
    assert [batch.next_cursor for batch in batches] == [2, 3]
    assert [len(batch.records) for batch in batches] == [2, 1]
    assert http.calls == [
        (
            "/occurrence/search",
            {
                "taxonKey": "1938069",
                "hasCoordinate": "true",
                "limit": 2,
                "offset": 0,
            },
        ),
        (
            "/occurrence/search",
            {
                "taxonKey": "1938069",
                "hasCoordinate": "true",
                "limit": 2,
                "offset": 2,
            },
        ),
    ]


def test_search_source_requires_bulk_download_above_documented_ceiling() -> None:
    http = FakeHTTPGet(
        {
            0: {
                "count": GBIF_SEARCH_MAX_RECORDS + 1,
                "offset": 0,
                "limit": 1,
                "endOfRecords": False,
                "results": [_occurrence("1")],
            }
        }
    )
    source = GBIFOccurrenceSearchSource(
        accepted_taxon_key=TARGET_KEY,
        source_snapshot_version=SNAPSHOT_VERSION,
        http_get=http,
        page_size=1,
    )

    with pytest.raises(BulkDownloadRequired, match="100000") as raised:
        tuple(source.iter_batches())

    assert raised.value.request_payload == build_gbif_bulk_download_request(
        accepted_taxon_key=TARGET_KEY
    )
    assert raised.value.request_payload["format"] == "SIMPLE_PARQUET"


def test_search_source_shrinks_the_page_at_the_documented_ceiling() -> None:
    http = FakeHTTPGet(
        {
            GBIF_SEARCH_MAX_RECORDS - 1: {
                "count": GBIF_SEARCH_MAX_RECORDS,
                "offset": GBIF_SEARCH_MAX_RECORDS - 1,
                "limit": 1,
                "endOfRecords": True,
                "results": [_occurrence("last")],
            }
        }
    )
    source = GBIFOccurrenceSearchSource(
        accepted_taxon_key=TARGET_KEY,
        source_snapshot_version=SNAPSHOT_VERSION,
        http_get=http,
    )

    batches = tuple(source.iter_batches(start_cursor=GBIF_SEARCH_MAX_RECORDS - 1))

    assert batches[0].next_cursor == GBIF_SEARCH_MAX_RECORDS
    assert http.calls[0][1]["limit"] == 1


def test_bulk_parquet_source_resumes_inside_a_batch(tmp_path: Path) -> None:
    path = tmp_path / "gbif-download.parquet"
    pl.DataFrame([_occurrence(str(index)) for index in range(3)]).write_parquet(path)
    source = GBIFParquetOccurrenceSource(
        path,
        accepted_taxon_key=TARGET_KEY,
        source_snapshot_version="gbif-download:000001",
        batch_size=2,
    )

    batches = tuple(source.iter_batches(start_cursor=1))

    assert [batch.cursor for batch in batches] == [1, 2]
    assert [batch.next_cursor for batch in batches] == [2, 3]
    assert [batch.end_of_records for batch in batches] == [False, True]


def test_species_key_is_authoritative_for_infraspecific_occurrences(tmp_path: Path) -> None:
    malformed_species_key = _occurrence("malformed", accepted_taxon_key=1_938_069)
    malformed_species_key["speciesKey"] = "not-a-key"
    records = (
        _occurrence("included", accepted_taxon_key=9_999, species_key=1_938_069),
        _occurrence("excluded", accepted_taxon_key=1_938_069, species_key=9_999),
        malformed_species_key,
    )
    source = FakeBatchSource(
        (
            OccurrenceBatch(
                cursor=0,
                next_cursor=3,
                records=records,
                end_of_records=True,
                total_records=3,
            ),
        )
    )

    result = build_taxon_geographic_spread(
        accepted_taxon_key=TARGET_KEY,
        scientific_name="Papilio demoleus",
        registry_version="butterflies-v1",
        source=source,
        resolutions=GeographicResolutions(coarse=3, regional=5, local=7),
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoint",
        retrieved_at=RETRIEVED_AT,
    )

    assert result.spread["occurrence_count"].to_list() == [1, 1, 1]
    assert result.manifest["taxon_key_mismatch_count"] == 2
    evidence = pl.read_parquet(result.evidence_path)
    matched_ids = set(evidence.filter("taxon_key_match")["gbif_id"].to_list())
    assert matched_ids == {"included"}


def test_conflicting_admin_labels_do_not_split_the_spread_primary_key(tmp_path: Path) -> None:
    records = (
        _occurrence("1", country_code="AU", admin1="Queensland"),
        _occurrence("2", country_code="ZZ", admin1="Conflicting label"),
    )
    source = FakeBatchSource(
        (
            OccurrenceBatch(
                cursor=0,
                next_cursor=2,
                records=records,
                end_of_records=True,
                total_records=2,
            ),
        )
    )

    result = build_taxon_geographic_spread(
        accepted_taxon_key=TARGET_KEY,
        scientific_name="Papilio demoleus",
        registry_version="butterflies-v1",
        source=source,
        resolutions=GeographicResolutions(coarse=3, regional=5, local=7),
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoint",
        retrieved_at=RETRIEVED_AT,
    )

    assert result.spread.height == 3
    assert result.spread["occurrence_count"].to_list() == [2, 2, 2]
    assert result.spread["country_code"].to_list() == [None, None, None]
    assert result.spread["admin1"].to_list() == [None, None, None]


def test_spread_compilation_retains_provenance_and_separates_ineligible_records(
    tmp_path: Path,
) -> None:
    records = (
        _occurrence(
            "1",
            basis_of_record="HUMAN_OBSERVATION",
            uncertainty=10.0,
            event_date="2024-01-02T03:04:05Z",
        ),
        _occurrence(
            "2",
            basis_of_record="PRESERVED_SPECIMEN",
            uncertainty=100.0,
            event_date="1990-05-01",
        ),
        _occurrence(
            "3",
            basis_of_record="FOSSIL_SPECIMEN",
            uncertainty=None,
            event_date="1900-01-01",
        ),
        _occurrence(
            "4",
            basis_of_record="HUMAN_OBSERVATION",
            uncertainty=50.0,
            event_date="2025-06-07",
            issues=["COORDINATE_INVALID"],
        ),
    )
    source = FakeBatchSource(
        (
            OccurrenceBatch(
                cursor=0,
                next_cursor=4,
                records=records,
                end_of_records=True,
                total_records=4,
            ),
        )
    )

    result = build_taxon_geographic_spread(
        accepted_taxon_key=TARGET_KEY,
        scientific_name="Papilio demoleus",
        registry_version="butterflies-v1",
        source=source,
        resolutions=GeographicResolutions(coarse=3, regional=5, local=7),
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoint",
        retrieved_at=RETRIEVED_AT,
    )

    assert result.spread.schema == geographic_spread_schema()
    assert result.spread["schema_version"].unique().to_list() == [
        GEOGRAPHIC_SPREAD_SCHEMA_VERSION
    ]
    assert result.spread.height == 3
    assert result.spread["spatial_resolution"].to_list() == [3, 5, 7]
    row = result.spread.row(0, named=True)
    assert row["accepted_taxon_key"] == TARGET_KEY
    assert row["gbif_species_key"] == 1938069
    assert row["source_dataset_key"] == "dataset-1"
    assert row["source_dataset_citation"] == "Fixture butterfly observations"
    assert row["occurrence_count"] == 4
    assert row["georeferenced_occurrence_count"] == 4
    assert row["range_inference_eligible_count"] == 1
    assert row["preserved_specimen_count"] == 1
    assert row["fossil_count"] == 1
    assert row["geospatial_issue_count"] == 1
    assert row["earliest_occurrence_date"].isoformat() == "1900-01-01"
    assert row["latest_occurrence_date"].isoformat() == "2025-06-07"
    assert row["basis_of_record_counts"] == [
        {"value": "FOSSIL_SPECIMEN", "count": 1},
        {"value": "HUMAN_OBSERVATION", "count": 2},
        {"value": "PRESERVED_SPECIMEN", "count": 1},
    ]
    assert row["coordinate_uncertainty_summary"] == {
        "count": 3,
        "min_m": 10.0,
        "p50_m": 50.0,
        "p95_m": 100.0,
        "max_m": 100.0,
    }
    assert row["known_range_role"] == "native"
    assert row["evidence_confidence"] == pytest.approx(0.25)
    assert row["source_snapshot_version"] == SNAPSHOT_VERSION
    assert (tmp_path / "output" / TAXON_GEOGRAPHIC_SPREAD_FILE).exists()
    assert result.manifest["invalid_coordinate_count"] == 0
    assert result.manifest["taxon_key_mismatch_count"] == 0
    assert result.manifest["range_inference_eligible_occurrence_count"] == 1


def test_invalid_coordinates_and_taxon_mismatches_are_checkpointed_not_aggregated(
    tmp_path: Path,
) -> None:
    invalid = _occurrence("invalid", latitude=91.0)
    mismatch = _occurrence("mismatch", accepted_taxon_key=999)
    source = FakeBatchSource(
        (
            OccurrenceBatch(
                cursor=0,
                next_cursor=2,
                records=(invalid, mismatch),
                end_of_records=True,
                total_records=2,
            ),
        )
    )

    result = build_taxon_geographic_spread(
        accepted_taxon_key=TARGET_KEY,
        scientific_name="Papilio demoleus",
        registry_version="butterflies-v1",
        source=source,
        resolutions=GeographicResolutions(coarse=3, regional=5, local=7),
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoint",
        retrieved_at=RETRIEVED_AT,
    )

    assert result.spread.is_empty()
    assert result.manifest["invalid_coordinate_count"] == 1
    assert result.manifest["taxon_key_mismatch_count"] == 1
    evidence = pl.read_parquet(result.evidence_path)
    assert evidence.height == 2
    assert sorted(evidence["exclusion_reason"].to_list()) == [
        "invalid_coordinate",
        "taxon_key_mismatch",
    ]


def test_interrupted_build_resumes_from_validated_cursor(tmp_path: Path) -> None:
    batches = (
        OccurrenceBatch(
            cursor=0,
            next_cursor=1,
            records=(_occurrence("1"),),
            end_of_records=False,
            total_records=2,
        ),
        OccurrenceBatch(
            cursor=1,
            next_cursor=2,
            records=(_occurrence("2"),),
            end_of_records=True,
            total_records=2,
        ),
    )
    interrupted = FakeBatchSource(batches, fail_after_batches=1)
    kwargs = {
        "accepted_taxon_key": TARGET_KEY,
        "scientific_name": "Papilio demoleus",
        "registry_version": "butterflies-v1",
        "resolutions": GeographicResolutions(coarse=3, regional=5, local=7),
        "output_dir": tmp_path / "output",
        "checkpoint_dir": tmp_path / "checkpoint",
        "retrieved_at": RETRIEVED_AT,
    }

    with pytest.raises(RuntimeError, match="injected source interruption"):
        build_taxon_geographic_spread(source=interrupted, **kwargs)

    resumed_source = FakeBatchSource(batches)
    result = build_taxon_geographic_spread(source=resumed_source, **kwargs)

    assert resumed_source.start_cursors == [1]
    assert result.resumed is True
    assert result.manifest["completed_occurrence_count"] == 2
    assert result.manifest["checkpoint_part_count"] == 2
    assert result.spread["occurrence_count"].to_list() == [2, 2, 2]


def test_resume_rejects_changed_snapshot_identity(tmp_path: Path) -> None:
    batches = (
        OccurrenceBatch(
            cursor=0,
            next_cursor=1,
            records=(_occurrence("1"),),
            end_of_records=False,
            total_records=2,
        ),
        OccurrenceBatch(
            cursor=1,
            next_cursor=2,
            records=(_occurrence("2"),),
            end_of_records=True,
            total_records=2,
        ),
    )
    interrupted = FakeBatchSource(batches, fail_after_batches=1)
    kwargs = {
        "accepted_taxon_key": TARGET_KEY,
        "scientific_name": "Papilio demoleus",
        "registry_version": "butterflies-v1",
        "resolutions": GeographicResolutions(coarse=3, regional=5, local=7),
        "output_dir": tmp_path / "output",
        "checkpoint_dir": tmp_path / "checkpoint",
        "retrieved_at": RETRIEVED_AT,
    }
    with pytest.raises(RuntimeError):
        build_taxon_geographic_spread(source=interrupted, **kwargs)

    changed = FakeBatchSource(batches)
    changed.source_snapshot_version = "different-snapshot"
    with pytest.raises(ValueError, match="source_snapshot_version"):
        build_taxon_geographic_spread(source=changed, **kwargs)


def test_resume_adopts_an_atomic_part_written_before_checkpoint_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from biominer.registry import geographic_spread as spread_module

    batch = OccurrenceBatch(
        cursor=0,
        next_cursor=1,
        records=(_occurrence("1"),),
        end_of_records=True,
        total_records=1,
    )
    kwargs = {
        "accepted_taxon_key": TARGET_KEY,
        "scientific_name": "Papilio demoleus",
        "registry_version": "butterflies-v1",
        "resolutions": GeographicResolutions(coarse=3, regional=5, local=7),
        "output_dir": tmp_path / "output",
        "checkpoint_dir": tmp_path / "checkpoint",
        "retrieved_at": RETRIEVED_AT,
    }
    write_state = spread_module._write_json_atomic

    def interrupt_state_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected state-write interruption")

    monkeypatch.setattr(spread_module, "_write_json_atomic", interrupt_state_write)
    with pytest.raises(RuntimeError, match="state-write interruption"):
        build_taxon_geographic_spread(source=FakeBatchSource((batch,)), **kwargs)

    monkeypatch.setattr(spread_module, "_write_json_atomic", write_state)
    result = build_taxon_geographic_spread(source=FakeBatchSource((batch,)), **kwargs)

    assert result.resumed is True
    assert result.manifest["completed_occurrence_count"] == 1
    assert result.spread["occurrence_count"].to_list() == [1, 1, 1]


def _occurrence(
    gbif_id: str,
    *,
    accepted_taxon_key: int = 1938069,
    species_key: int | None = None,
    latitude: float = -27.4705,
    longitude: float = 153.0260,
    basis_of_record: str = "HUMAN_OBSERVATION",
    uncertainty: float | None = 25.0,
    event_date: str = "2025-01-01",
    issues: list[str] | None = None,
    country_code: str = "AU",
    admin1: str = "Queensland",
) -> dict[str, object]:
    row: dict[str, object] = {
        "key": gbif_id,
        "acceptedTaxonKey": accepted_taxon_key,
        "acceptedScientificName": "Papilio demoleus",
        "decimalLatitude": latitude,
        "decimalLongitude": longitude,
        "coordinateUncertaintyInMeters": uncertainty,
        "countryCode": country_code,
        "stateProvince": admin1,
        "datasetKey": "dataset-1",
        "datasetTitle": "Fixture butterfly observations",
        "basisOfRecord": basis_of_record,
        "establishmentMeans": "NATIVE",
        "occurrenceStatus": "PRESENT",
        "eventDate": event_date,
        "issues": issues or [],
    }
    if species_key is not None:
        row["speciesKey"] = species_key
    return row
