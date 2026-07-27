from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.representativeness import publish_representativeness


def test_representativeness_keeps_raw_and_duplicate_adjusted_counts(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    quality = tmp_path / "quality.parquet"
    readiness = tmp_path / "readiness"
    readiness.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                _source("1", "Species one", "Provider A", "Creator A", "2020"),
                _source("2", "Species one", "Provider A", "Creator A", "2021"),
                _source("3", "Species two", "Provider B", None, "2021"),
            ]
        ),
        source,
    )
    pq.write_table(pa.table({"media_assertion_id": ["m1", "m2", "m3"]}), quality)
    pq.write_table(
        pa.Table.from_pylist(
            [
                _readiness("m1", "1", "u1", "PASS", "NOT_TESTED"),
                _readiness("m2", "2", "u1", "PASS", "NOT_TESTED"),
                _readiness("m3", "3", "u3", "UNKNOWN", "UNRESOLVED"),
            ]
        ),
        readiness / "part-0.parquet",
    )
    manifest = publish_representativeness(
        v3_parquet=source,
        media_quality_parquet=quality,
        ai_readiness_glob=readiness / "*.parquet",
        output_directory=tmp_path / "out",
        source_snapshot_id="snapshot",
        expected_rows=3,
        code_commit="deadbeef",
        threads=1,
    )
    coverage = pq.read_table(tmp_path / "out/coverage_by_dimension.parquet").to_pylist()
    species_one = next(
        row for row in coverage if row["dimension"] == "species" and row["value"] == "Species one"
    )
    assert species_one["raw_image_count"] == 2
    assert species_one["duplicate_adjusted_count"] == 1
    flags = pq.read_table(tmp_path / "out/species_bias_flags.parquet").to_pylist()
    assert "ONE_PROVIDER" in next(row for row in flags if row["species"] == "Species one")["bias_flags"]
    assert manifest["counts"]["source_media_rows"] == 3
    assert manifest["configuration"]["composite_quality_score"] is False


def _source(gbif_id, species, provider, creator, year):
    return {
        "gbifID": gbif_id,
        "kingdom": "Animalia",
        "phylum": "Arthropoda",
        "class": "Insecta",
        "order": "Lepidoptera",
        "family": "Family",
        "genus": species.split()[0],
        "species": species,
        "taxonRank": "SPECIES",
        "acceptedTaxonKey": species,
        "media_publisher": provider,
        "publisher": provider,
        "datasetName": "Dataset",
        "datasetKey": "dataset-key",
        "media_creator": creator,
        "countryCode": "AU",
        "continent": "OCEANIA",
        "gbifRegion": "OCEANIA",
        "decimalLatitude": "-33.8",
        "decimalLongitude": "151.2",
        "year": year,
        "month": "1",
        "basisOfRecord": "HUMAN_OBSERVATION",
        "lifeStage": None,
        "sex": None,
        "media_license": "CC BY 4.0",
        "media_format": "image/jpeg",
        "media_identifier": f"https://example.org/{gbif_id}.jpg",
        "media_rightsHolder": creator,
        "coordinateUncertaintyInMeters": "10",
    }


def _readiness(media_id, gbif_id, url_hash, rights, decision):
    return {
        "media_assertion_id": media_id,
        "gbifID": gbif_id,
        "original_url_hash": url_hash,
        "canonical_url_hash": url_hash,
        "rights_policy_status": "ALLOWED" if rights == "PASS" else "QUARANTINED",
        "duplicate_status": "DUPLICATE" if url_hash == "u1" else "PASS",
        "cross_taxon_url_status": "PASS",
        "MEDIA_DIRECT": "PASS",
        "RIGHTS_KNOWN": rights,
        "RIGHTS_ALLOWED": rights,
        "MEDIA_TECHNICALLY_VALID": "NOT_TESTED",
        "EXACT_SPECIES_LABEL": "PASS",
        "IDENTIFICATION_PROVENANCE_PRESENT": "PASS",
        "ai_ingestion_decision": decision,
    }
