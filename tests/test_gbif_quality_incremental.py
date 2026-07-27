from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.incremental import publish_incremental_state


def test_incremental_state_does_not_queue_unchanged_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    quality = tmp_path / "quality.parquet"
    duplicates = tmp_path / "duplicates.parquet"
    pq.write_table(pa.Table.from_pylist([_source("1"), _source("2")]), source)
    pq.write_table(
        pa.table(
            {
                "source_row_id": ["r1", "r2"],
                "media_assertion_id": ["m1", "m2"],
            }
        ),
        quality,
    )
    pq.write_table(
        pa.table(
            {
                "media_assertion_id": ["m1", "m2"],
                "source_value_hash": ["sha256:" + "01" * 32, "sha256:" + "02" * 32],
            }
        ),
        duplicates,
    )
    baseline = publish_incremental_state(
        v3_parquet=source,
        media_quality_parquet=quality,
        duplicates_parquet=duplicates,
        output_directory=tmp_path / "baseline",
        source_snapshot_id="snapshot",
        expected_rows=2,
        code_commit="deadbeef",
        threads=1,
        partitions=2,
    )
    assert baseline["counts"]["queue_rows"] == 0
    rerun = publish_incremental_state(
        v3_parquet=source,
        media_quality_parquet=quality,
        duplicates_parquet=duplicates,
        output_directory=tmp_path / "rerun",
        source_snapshot_id="snapshot",
        expected_rows=2,
        code_commit="deadbeef",
        previous_state_glob=tmp_path / "baseline/state/**/*.parquet",
        threads=1,
        partitions=2,
    )
    assert rerun["counts"]["queue_rows"] == 0
    assert (
        rerun["counts"]["state_semantic_fingerprint"]
        == rerun["counts"]["previous_semantic_fingerprint"]
    )
    assert rerun["validation"]["unchanged_rerun_semantically_identical"] is True
    changed_rows = [_source("1"), _source("2")]
    changed_rows[1]["media_identifier"] = "https://example.org/2-updated.jpg"
    pq.write_table(pa.Table.from_pylist(changed_rows), source)
    changed = publish_incremental_state(
        v3_parquet=source,
        media_quality_parquet=quality,
        duplicates_parquet=duplicates,
        output_directory=tmp_path / "changed",
        source_snapshot_id="snapshot-2",
        expected_rows=2,
        code_commit="deadbeef",
        previous_state_glob=tmp_path / "baseline/state/**/*.parquet",
        threads=1,
        partitions=2,
    )
    assert changed["counts"]["queue_rows"] == 1
    queue = pq.read_table(tmp_path / "changed/changed_row_queue.parquet").to_pylist()
    assert queue[0]["change_status"] == "CHANGED"
    assert "MEDIA_URL_CHANGED" in queue[0]["change_reasons"]


def _source(gbif_id: str) -> dict[str, object]:
    return {
        "gbifID": gbif_id,
        "media_identifier": f"https://example.org/{gbif_id}.jpg",
        "media_references": None,
        "media_license": "CC BY 4.0",
        "media_creator": "Creator",
        "media_rightsHolder": "Holder",
        "decimalLatitude": "-33.8",
        "decimalLongitude": "151.2",
        "coordinateUncertaintyInMeters": "10",
        "eventDate": "2025-01-02",
        "year": "2025",
        "month": "1",
        "day": "2",
        "identifiedBy": "Identifier",
        "identificationVerificationStatus": "accepted",
        "dateIdentified": "2025-01-03",
        "scientificName": "Species one",
        "taxonRank": "SPECIES",
        "taxonKey": "1",
        "acceptedTaxonKey": "1",
        "taxonomicStatus": "ACCEPTED",
        "datasetKey": "dataset",
        "publisher": "Publisher",
        "media_publisher": "Provider",
    }
