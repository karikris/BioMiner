from __future__ import annotations

import duckdb
import polars as pl

from flickr_bio_occurrence.dwc.exporter import EXPORT_CSV_BY_DEFAULT, export_dwc_records
from flickr_bio_occurrence.storage.duckdb_index import create_qa_views
from flickr_bio_occurrence.storage.parquet_io import write_parquet_dataset


def test_write_parquet_dataset_creates_partitioned_parquet(tmp_path) -> None:
    frame = pl.DataFrame(
        {
            "scientificName": ["Papilio demoleus"],
            "region_id": ["AU_QLD"],
            "year": [2024],
            "month": [1],
        }
    )

    written = write_parquet_dataset(frame, tmp_path / "bronze", partition_by=["scientificName", "region_id", "year", "month"])

    assert len(written) == 1
    assert written[0].suffix == ".parquet"
    assert "scientificName=Papilio demoleus" in str(written[0])


def test_dwc_export_defaults_to_parquet_only(tmp_path) -> None:
    frame = pl.DataFrame({"occurrenceID": ["abc"], "scientificName": ["Papilio demoleus"]})

    outputs = export_dwc_records(frame, tmp_path / "gold")

    assert EXPORT_CSV_BY_DEFAULT is False
    assert outputs.parquet_paths
    assert outputs.csv_path is None
    assert not list(tmp_path.rglob("*.csv"))


def test_duckdb_qa_views_read_parquet_outputs(tmp_path) -> None:
    bronze = pl.DataFrame({"flickr_photo_id": ["1"], "scientificName": ["Papilio demoleus"]})
    silver = pl.DataFrame({"flickr_photo_id": ["1"], "review_status": ["needs_review"], "range_extension_candidate": [True]})
    gold = pl.DataFrame({"occurrenceID": ["abc"], "scientificName": ["Papilio demoleus"]})
    write_parquet_dataset(bronze, tmp_path / "bronze" / "bronze_flickr_photo")
    write_parquet_dataset(silver, tmp_path / "silver" / "silver_occurrence_candidate")
    write_parquet_dataset(gold, tmp_path / "gold" / "dwc_occurrence")

    db_path = tmp_path / "qa.duckdb"
    create_qa_views(db_path=db_path, data_root=tmp_path)

    with duckdb.connect(str(db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM raw_photos").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM occurrence_candidates WHERE review_status = 'needs_review'").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM dwc_occurrence").fetchone()[0] == 1
