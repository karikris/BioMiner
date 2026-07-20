from __future__ import annotations

import json
import zipfile

import polars as pl

from biominer.registry.gbif_occurrence_source import build_gbif_source_snapshot_from_occurrence_archive
from biominer.registry.scope import load_scope


def _write_scope(path) -> None:
    path.write_text(
        json.dumps(
            {
                "scope_id": "test-scope",
                "root": {"scientific_name": "Lepidoptera", "rank": "ORDER", "key": "1000"},
                "included_families": ["Nymphalidae"],
                "gbif_family_taxon_keys": {"Nymphalidae": 10},
            }
        ),
        encoding="utf-8",
    )


def _write_occurrence_archive(path) -> None:
    rows = [
        {
            "familyKey": "10",
            "family": "Nymphalidae",
            "acceptedTaxonKey": "100",
            "species": "Danaus plexippus",
            "scientificName": "Danaus plexippus",
            "genusKey": "90",
            "genus": "Danaus",
            "datasetKey": "dataset-1",
            "datasetName": "test-dataset",
            "countryCode": "US",
            "vernacularName": "Monarch",
            "language": "eng",
            "taxonomicStatus": "ACCEPTED",
        },
        {
            "familyKey": "999",
            "family": "Pieridae",
            "acceptedTaxonKey": "200",
            "species": "Unknown outside scope",
            "scientificName": "Pieris unknown",
            "datasetKey": "dataset-2",
            "datasetName": "other-dataset",
            "taxonomicStatus": "ACCEPTED",
        },
    ]
    frame = pl.DataFrame(rows)
    parquet_path = path.with_suffix(".parquet")
    frame.write_parquet(parquet_path)
    try:
        with zipfile.ZipFile(path, mode="w") as bundle:
            bundle.write(parquet_path, arcname="occurrence/occurrence.parquet")
    finally:
        parquet_path.unlink(missing_ok=True)


def test_build_gbif_source_snapshot_from_occurrence_archive_writes_parquet_and_deletes_download(tmp_path) -> None:
    scope_path = tmp_path / "scope.json"
    _write_scope(scope_path)
    scope = load_scope(scope_path)
    archive = tmp_path / "gbif-occurrence-download.zip"
    _write_occurrence_archive(archive)
    output_parquet = tmp_path / "gbif_occurrences.parquet"

    snapshot = build_gbif_source_snapshot_from_occurrence_archive(
        archive,
        scope,
        retrieved_at="2026-07-19T00:00:00+00:00",
        source_parquet=output_parquet,
    )

    assert snapshot["source"] == "GBIF"
    assert snapshot["source_dataset_key"] == "gbif:dataset-1"
    assert snapshot["source_dataset_citation"] == "test-dataset"
    assert snapshot["metrics"]["rows_in_scope"] == 1
    assert snapshot["metrics"]["rows_scanned"] == 2
    assert output_parquet.exists()
    parquet_rows = pl.read_parquet(output_parquet)
    assert parquet_rows.height == 1
    assert parquet_rows[0, "accepted_taxon_key"] == "gbif:100"
    assert not archive.exists()


def test_build_gbif_source_snapshot_default_output_name_is_archive_parquet(tmp_path) -> None:
    scope_path = tmp_path / "scope.json"
    _write_scope(scope_path)
    scope = load_scope(scope_path)
    archive = tmp_path / "gbif-occurrence-download.zip"
    _write_occurrence_archive(archive)
    expected_parquet = archive.with_suffix(".parquet")

    build_gbif_source_snapshot_from_occurrence_archive(
        archive,
        scope,
        retrieved_at="2026-07-19T00:00:00+00:00",
        delete_download_after=False,
    )

    assert expected_parquet.exists()
    assert archive.exists()
