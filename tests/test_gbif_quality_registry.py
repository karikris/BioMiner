from __future__ import annotations

from biominer.gbif_quality.registry import (
    CHECK_REGISTRY_VERSION,
    QualityStatus,
    check_registry,
    check_registry_table,
    registry_fingerprint,
)


def test_quality_status_model_is_complete_and_non_collapsing() -> None:
    assert {status.value for status in QualityStatus} == {
        "PASS",
        "FAIL",
        "UNKNOWN",
        "NOT_APPLICABLE",
        "WITHHELD",
        "GENERALIZED",
        "CONFLICT",
        "NOT_TESTED",
    }


def test_registry_has_stable_unique_checks_and_required_contract() -> None:
    checks = check_registry()
    identifiers = {check.check_id for check in checks}

    assert len(identifiers) == len(checks)
    assert {
        "SOURCE_001",
        "SCHEMA_001",
        "ID_001",
        "NULL_001",
        "MEDIA_URL_001",
        "MEDIA_FILE_001",
        "RIGHTS_001",
        "TIME_001",
        "GEO_001",
        "TAXON_001",
        "IDENT_001",
        "VOCAB_001",
        "PROVENANCE_001",
    } <= identifiers
    assert all(check.rule_version for check in checks)
    assert all(check.required_evidence for check in checks)
    assert all(not check.automatic_repair_permitted for check in checks)
    network = [check for check in checks if check.network_required]
    assert network and all(not check.deterministic for check in network)

    table = check_registry_table(checks)
    assert table.num_rows == len(checks)
    assert table["registry_version"].to_pylist() == [CHECK_REGISTRY_VERSION] * len(checks)
    assert len(set(table["registry_fingerprint"].to_pylist())) == 1
    assert table["registry_fingerprint"][0].as_py() == registry_fingerprint(checks)
