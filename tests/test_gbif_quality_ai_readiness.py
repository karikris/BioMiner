from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from biominer.gbif_quality.ai_readiness import (
    AI_READINESS_SCHEMA,
    publish_ai_readiness,
)


def test_ai_readiness_is_fail_closed_without_network_or_image_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    media_quality = tmp_path / "media_quality.parquet"
    occurrence_quality = tmp_path / "occurrence_quality.parquet"
    rights = tmp_path / "rights.parquet"
    duplicates = tmp_path / "duplicates.parquet"
    repairs = tmp_path / "repairs.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _source_row("1", "https://example.org/1.jpg", None, "SPECIES", "A a"),
                _source_row("2", None, "https://example.org/occ/2", "GENUS", None),
                _source_row("3", "https://example.org/3.jpg", None, "SPECIES", "B b"),
            ]
        ),
        source,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _media_quality("1", "r1", "m1", "PASS", "NOT_APPLICABLE"),
                _media_quality("2", "r2", "m2", "UNKNOWN", "PASS"),
                _media_quality("3", "r3", "m3", "PASS", "NOT_APPLICABLE"),
            ]
        ),
        media_quality,
    )
    pq.write_table(
        pa.Table.from_pylist([_occurrence_quality(str(index)) for index in range(1, 4)]),
        occurrence_quality,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _rights("1", "r1", "m1", "ALLOWED", "PASS"),
                _rights("2", "r2", "m2", "QUARANTINED", "UNKNOWN"),
                _rights("3", "r3", "m3", "DENIED", "PASS"),
            ]
        ),
        rights,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _duplicate("1", "r1", "m1", "PASS"),
                _duplicate("2", "r2", "m2", "PASS"),
                _duplicate("3", "r3", "m3", "CONFLICT"),
            ]
        ),
        duplicates,
    )
    pq.write_table(
        pa.table(
            {
                "gbifID": pa.array([], type=pa.string()),
                "derived_species": pa.array([], type=pa.string()),
                "derivation_status": pa.array([], type=pa.string()),
            }
        ),
        repairs,
    )

    manifest = publish_ai_readiness(
        v3_parquet=source,
        media_quality_parquet=media_quality,
        occurrence_quality_parquet=occurrence_quality,
        rights_parquet=rights,
        duplicates_parquet=duplicates,
        taxonomy_repairs_parquet=repairs,
        output_directory=tmp_path / "out",
        source_snapshot_id="snapshot",
        expected_rows=3,
        code_commit="deadbeef",
        threads=1,
        part_size="1MB",
    )

    table = ds.dataset(tmp_path / "out/parts", format="parquet").to_table()
    assert table.schema.equals(AI_READINESS_SCHEMA)
    rows = sorted(table.to_pylist(), key=lambda row: row["gbifID"])
    assert rows[0]["MEDIA_ADDRESSABLE"] == "PASS"
    assert rows[0]["MEDIA_TECHNICALLY_VALID"] == "NOT_TESTED"
    assert rows[0]["AI_CLASSIFICATION_READY"] == "NOT_TESTED"
    assert rows[0]["ai_ingestion_decision"] == "NOT_TESTED"
    assert "IMAGE_BYTES_NOT_INSPECTED" in rows[0]["reason_codes"]
    assert rows[1]["ai_ingestion_decision"] == "UNRESOLVED"
    assert rows[1]["RIGHTS_KNOWN"] == "UNKNOWN"
    assert rows[2]["EXCLUDED"] == "PASS"
    assert rows[2]["ai_ingestion_decision"] == "EXCLUDED"
    assert manifest["network_requests"] == 0
    assert manifest["counts"]["rows"] == 3


def _source_row(gbif_id, direct, reference, rank, species):
    return {
        "gbifID": gbif_id,
        "media_identifier": direct,
        "media_references": reference,
        "media_type": "StillImage",
        "media_format": "image/jpeg" if direct else None,
        "countryCode": "AU",
        "species": species,
        "scientificName": species or "Genus",
        "taxonRank": rank,
        "datasetKey": "dataset",
        "decimalLatitude": "-33.8",
        "decimalLongitude": "151.2",
        "coordinateUncertaintyInMeters": "10",
        "eventID": None,
        "parentEventID": None,
        "locationID": None,
        "locality": "Sydney",
        "eventDate": "2025-01-02",
    }


def _media_quality(gbif_id, source_row_id, media_id, direct, reference):
    return {
        "source_row_id": source_row_id,
        "media_assertion_id": media_id,
        "gbifID": gbif_id,
        "direct_media_url_status": direct,
        "media_reference_url_status": reference,
    }


def _occurrence_quality(gbif_id):
    return {
        "gbifID": gbif_id,
        "gbif_id_status": "PASS",
        "basis_of_record_status": "PASS",
        "event_date_status": "PASS",
        "coordinate_pair_status": "PASS",
        "zero_coordinate_status": "PASS",
        "coordinate_uncertainty_status": "PASS",
        "rank_name_consistency_status": "PASS",
        "accepted_taxon_key_status": "PASS",
        "identified_by_status": "UNKNOWN",
        "verification_source_evidence_status": "UNKNOWN",
    }


def _rights(gbif_id, source_row_id, media_id, policy, normalization):
    return {
        "source_row_id": source_row_id,
        "media_assertion_id": media_id,
        "gbifID": gbif_id,
        "rights_policy_status": policy,
        "license_normalization_status": normalization,
        "attribution_status": "PASS",
    }


def _duplicate(gbif_id, source_row_id, media_id, cross_taxon):
    return {
        "source_row_id": source_row_id,
        "media_assertion_id": media_id,
        "gbifID": gbif_id,
        "original_url_hash": f"original-{media_id}",
        "canonical_url_hash": f"canonical-{media_id}",
        "occurrence_leakage_group_id": f"occ-{gbif_id}",
        "dataset_occurrence_leakage_group_id": f"dataset-occ-{gbif_id}",
        "creator_leakage_group_id": "creator",
        "source_platform_group_id": "platform",
        "duplicate_status": "CONFLICT" if cross_taxon == "CONFLICT" else "PASS",
        "cross_taxon_url_status": cross_taxon,
    }
