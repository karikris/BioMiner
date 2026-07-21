from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import polars as pl
import pytest

from biominer.registry.gbif_reference_media import (
    GBIFReferenceMediaManifestConfig,
    REFERENCE_MEDIA_MANIFEST_SCHEMA_VERSION,
    build_gbif_reference_media_manifest,
)


OCCURRENCE_COLUMNS = (
    "gbifID",
    "occurrenceID",
    "datasetKey",
    "datasetName",
    "publisher",
    "basisOfRecord",
    "lifeStage",
    "taxonKey",
    "acceptedTaxonKey",
    "speciesKey",
    "scientificName",
    "acceptedScientificName",
    "species",
    "genus",
    "family",
    "taxonRank",
    "taxonomicStatus",
)
MULTIMEDIA_COLUMNS = (
    "gbifID",
    "type",
    "format",
    "identifier",
    "references",
    "title",
    "description",
    "source",
    "audience",
    "created",
    "creator",
    "contributor",
    "publisher",
    "license",
    "rightsHolder",
)


def _row(columns: tuple[str, ...], **values: str) -> tuple[str, ...]:
    return tuple(values.get(column, "") for column in columns)


def _write_member(columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    return "\t".join(columns) + "\n" + "\n".join("\t".join(row) for row in rows) + "\n"


def _write_dwca(path: Path, *, duplicate_occurrence: bool = False) -> None:
    occurrences = [
        _row(
            OCCURRENCE_COLUMNS,
            gbifID="100",
            occurrenceID="occurrence-100",
            datasetKey="dataset-key-1",
            datasetName="Butterflies of Test Island",
            publisher="Occurrence Publisher",
            basisOfRecord="HUMAN_OBSERVATION",
            lifeStage="caterpillar",
            taxonKey="5100001",
            acceptedTaxonKey="5100000",
            speciesKey="5100000",
            scientificName="Papilio example synonym",
            acceptedScientificName="Papilio example Linnaeus",
            species="Papilio example",
            genus="Papilio",
            family="Papilionidae",
            taxonRank="SPECIES",
            taxonomicStatus="SYNONYM",
        ),
        _row(
            OCCURRENCE_COLUMNS,
            gbifID="101",
            occurrenceID="occurrence-101",
            datasetKey="dataset-key-2",
            datasetName="Second dataset",
            publisher="Second Publisher",
            basisOfRecord="MACHINE_OBSERVATION",
            lifeStage="adult",
            taxonKey="5200000",
            acceptedTaxonKey="5200000",
            speciesKey="5200000",
            scientificName="Danaus example",
            acceptedScientificName="Danaus example",
            species="Danaus example",
            genus="Danaus",
            family="Nymphalidae",
            taxonRank="SPECIES",
            taxonomicStatus="ACCEPTED",
        ),
    ]
    if duplicate_occurrence:
        occurrences.append(occurrences[0])
    media = [
        _row(
            MULTIMEDIA_COLUMNS,
            gbifID="100",
            type="StillImage",
            format="image/jpeg",
            identifier="https://images.example.test/full/100.jpg?token=kept",
            references="https://records.example.test/100",
            title="Caterpillar reference",
            description="Provider description",
            source="Provider source",
            audience="research",
            created="2024-01-02",
            creator="Photographer One",
            contributor="Contributor One",
            publisher="Media Publisher",
            license="https://creativecommons.org/licenses/by/4.0/",
            rightsHolder="Rights Holder One",
        ),
        _row(
            MULTIMEDIA_COLUMNS,
            gbifID="101",
            type="StillImage",
            format="image/png",
            identifier="http://images.example.test/101.png",
            license="CC0 1.0",
            rightsHolder="Rights Holder Two",
        ),
        _row(
            MULTIMEDIA_COLUMNS,
            gbifID="999",
            type="Sound",
            format="audio/mpeg",
            identifier="https://media.example.test/unjoined.mp3",
            publisher="Unjoined Publisher",
            license="custom licence",
            rightsHolder="Unjoined Rights Holder",
        ),
    ]
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("occurrence.txt", _write_member(OCCURRENCE_COLUMNS, occurrences))
        archive.writestr("multimedia.txt", _write_member(MULTIMEDIA_COLUMNS, media))


def _config(tmp_path: Path, archive: Path, *, occurrence_rows: int = 2) -> GBIFReferenceMediaManifestConfig:
    return GBIFReferenceMediaManifestConfig(
        archive=archive,
        output=tmp_path / "reference_media_manifest.parquet",
        receipt=tmp_path / "reference_media_manifest.json",
        download_key="0000001-test",
        download_doi="https://doi.org/10.15468/dl.test",
        download_url="https://api.gbif.org/v1/occurrence/download/request/test.zip",
        citation="GBIF.org test download",
        source_snapshot_version="gbif-test-snapshot",
        report_dir=tmp_path / "reports",
        expected_occurrence_rows=occurrence_rows,
        expected_multimedia_rows=3,
        csv_block_size=256,
        progress_interval=100,
        duckdb_threads=1,
        duckdb_memory_limit="1GB",
        temp_dir=tmp_path / "temporary",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_provenance_complete_manifest_and_preserves_unjoined_media(tmp_path: Path) -> None:
    # Arrange
    archive = tmp_path / "download.zip"
    _write_dwca(archive)
    config = _config(tmp_path, archive)

    # Act
    receipt = build_gbif_reference_media_manifest(config)

    # Assert
    frame = pl.read_parquet(config.output)
    assert frame.height == 3
    assert frame.get_column("source_media_row_number").to_list() == [1, 2, 3]
    assert frame.get_column("gbif_id").to_list() == ["100", "101", "999"]
    assert frame.get_column("occurrence_joined").to_list() == [True, True, False]
    expected_first_row_values = {
        "schema_version": REFERENCE_MEDIA_MANIFEST_SCHEMA_VERSION,
        "source": "GBIF",
        "source_download_key": "0000001-test",
        "source_download_doi": "https://doi.org/10.15468/dl.test",
        "occurrence_id": "occurrence-100",
        "dataset_key": "dataset-key-1",
        "dataset_name": "Butterflies of Test Island",
        "occurrence_publisher": "Occurrence Publisher",
        "provider_life_stage": "caterpillar",
        "taxon_key": "5100001",
        "accepted_taxon_key": "5100000",
        "species_key": "5100000",
        "image_url": "https://images.example.test/full/100.jpg?token=kept",
        "media_format": "image/jpeg",
        "media_license": "https://creativecommons.org/licenses/by/4.0/",
        "media_rights_holder": "Rights Holder One",
        "media_publisher": "Media Publisher",
        "media_is_still_image": True,
        "image_url_is_http": True,
    }
    first_row = frame.row(0, named=True)
    assert {
        column: first_row[column] for column in expected_first_row_values
    } == expected_first_row_values
    unjoined = frame.row(2, named=True)
    assert unjoined["occurrence_id"] is None
    assert unjoined["media_publisher"] == "Unjoined Publisher"
    assert unjoined["media_is_still_image"] is False
    assert frame.get_column("source_archive_sha256").n_unique() == 1
    assert frame.get_column("source_archive_sha256").item(0) == f"sha256:{_sha256(archive)}"
    assert frame.get_column("source_row_fingerprint").str.starts_with("sha256:").all()
    assert frame.get_column("source_row_fingerprint").n_unique() == 3

    saved_receipt = json.loads(config.receipt.read_text(encoding="utf-8"))
    assert saved_receipt == receipt
    assert receipt["input_rows"] == {"occurrence": 2, "multimedia": 3}
    assert receipt["join"]["joined_rows"] == 2
    assert receipt["join"]["unjoined_rows"] == 1
    assert receipt["output"]["sha256"] == f"sha256:{_sha256(config.output)}"
    assert receipt["metrics"]["rows_in"] == 5
    assert receipt["metrics"]["rows_out"] == 3
    assert receipt["metrics"]["mps_recommended_max_memory"] is None
    report_json = Path(receipt["reports"]["json"])
    report_markdown = Path(receipt["reports"]["markdown"])
    assert json.loads(report_json.read_text(encoding="utf-8")) == receipt
    assert "18,680,565" not in report_markdown.read_text(encoding="utf-8")
    assert "Output rows: 3" in report_markdown.read_text(encoding="utf-8")


def test_refuses_to_replace_an_immutable_manifest(tmp_path: Path) -> None:
    # Arrange
    archive = tmp_path / "download.zip"
    _write_dwca(archive)
    config = _config(tmp_path, archive)
    build_gbif_reference_media_manifest(config)
    original_parquet_hash = _sha256(config.output)
    original_receipt_hash = _sha256(config.receipt)

    # Act / Assert
    with pytest.raises(FileExistsError, match="immutable"):
        build_gbif_reference_media_manifest(config)

    assert _sha256(config.output) == original_parquet_hash
    assert _sha256(config.receipt) == original_receipt_hash


def test_duplicate_occurrence_ids_fail_without_publishing_outputs(tmp_path: Path) -> None:
    # Arrange
    archive = tmp_path / "download.zip"
    _write_dwca(archive, duplicate_occurrence=True)
    config = _config(tmp_path, archive, occurrence_rows=3)

    # Act / Assert
    with pytest.raises(ValueError, match="duplicate gbifID"):
        build_gbif_reference_media_manifest(config)

    assert not config.output.exists()
    assert not config.receipt.exists()
