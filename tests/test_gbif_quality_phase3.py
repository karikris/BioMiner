from biominer.gbif_quality.assertions import assertion_table, build_assertion
from biominer.gbif_quality.phase3 import semantic_assertion_fingerprint


def test_phase3_semantic_fingerprint_ignores_operational_timestamp() -> None:
    common = dict(
        source_snapshot_version="snapshot",
        source_row_id="row",
        gbif_id="1",
        target_field="derived_year",
        original_value=None,
        derived_value=2020,
        evidence_source="eventDate",
        derivation_method="iso",
        derivation_rule_version="v1",
        confidence_class="DETERMINISTIC_DERIVATION",
        validation_status="PASS",
        conflict_status="PASS",
        reviewer_status="NOT_REQUIRED",
    )
    first = assertion_table(
        [build_assertion(**common, retrieval_timestamp="2026-01-01T00:00:00Z")]
    )
    second = assertion_table(
        [build_assertion(**common, retrieval_timestamp="2026-02-01T00:00:00Z")]
    )

    assert semantic_assertion_fingerprint(first) == semantic_assertion_fingerprint(second)
