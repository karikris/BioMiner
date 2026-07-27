from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.taxonomy import (
    accepted_species_binomial,
    publish_species_rank_repairs,
)


def test_accepted_species_binomial_is_conservative() -> None:
    assert accepted_species_binomial("Pieris melete subsp. latouchei Mell, 1939") == "Pieris melete"
    assert accepted_species_binomial("Eurema alitha esakii Shirozu, 1953") == "Eurema alitha"
    assert accepted_species_binomial("Pieris sp.") is None
    assert accepted_species_binomial("pieris melete") is None
    assert accepted_species_binomial(None) is None


def test_species_rank_repairs_require_same_record_direct_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    rows = [
        _row("1", "SPECIES", None, "Pieris latouchei Mell, 1939", "Pieris melete subsp. latouchei Mell, 1939", "RGMQB", "RGNFJ"),
        _row("1", "SPECIES", None, "Pieris latouchei Mell, 1939", "Pieris melete subsp. latouchei Mell, 1939", "RGMQB", "RGNFJ"),
        _row("2", "SPECIES", None, "Unknown name", None, "ABC", None),
        _row("3", "GENUS", None, "Pieris", "Pieris", "XYZ", "XYZ"),
        _row("4", "SPECIES", "Pieris melete", "Pieris melete", "Pieris melete", "XYZ", "XYZ"),
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)

    result = publish_species_rank_repairs(
        v3_parquet=source,
        output_directory=tmp_path / "out",
        source_snapshot_id="sha256:test",
        expected_candidate_media_rows=3,
        expected_candidate_occurrences=2,
        code_commit="deadbeef",
    )

    repairs = {row["gbifID"]: row for row in pq.read_table(result.repair_path).to_pylist()}
    assert repairs["1"]["derived_species"] == "Pieris melete"
    assert repairs["1"]["derivation_status"] == "PASS"
    assert repairs["2"]["derived_species"] is None
    assert repairs["2"]["derivation_status"] == "UNKNOWN"
    assertions = pq.read_table(result.assertion_path).to_pylist()
    assert len(assertions) == 1
    assert assertions[0]["target_field"] == "derived_species"
    assert result.manifest["counts"]["repaired_media_rows"] == 2


def _row(
    gbif_id: str,
    rank: str,
    species: str | None,
    scientific_name: str | None,
    accepted_name: str | None,
    taxon_key: str | None,
    accepted_key: str | None,
) -> dict[str, str | None]:
    return {
        "gbifID": gbif_id,
        "taxonRank": rank,
        "species": species,
        "scientificName": scientific_name,
        "acceptedScientificName": accepted_name,
        "taxonKey": taxon_key,
        "acceptedTaxonKey": accepted_key,
        "taxonomicStatus": "SYNONYM",
    }
