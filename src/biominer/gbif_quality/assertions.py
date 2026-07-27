from __future__ import annotations

from dataclasses import dataclass
import hashlib

import pyarrow as pa


DERIVED_ASSERTION_VERSION = "biominer-gbif-derived-assertion/v1"
CONFIDENCE_CLASSES = frozenset(
    {
        "DIRECT_SOURCE",
        "DETERMINISTIC_DERIVATION",
        "PROVIDER_ASSERTION",
        "STRUCTURED_PAGE_METADATA",
        "CONTROLLED_TEXT_EXTRACTION",
        "MODEL_CANDIDATE",
        "MANUALLY_VERIFIED",
        "UNRESOLVED",
    }
)
DERIVED_ASSERTION_SCHEMA = pa.schema(
    [
        ("assertion_version", pa.string()),
        ("assertion_id", pa.string()),
        ("source_row_id", pa.string()),
        ("gbifID", pa.string()),
        ("media_assertion_id", pa.string()),
        ("target_field", pa.string()),
        ("original_value", pa.string()),
        ("derived_value", pa.string()),
        ("evidence_source", pa.string()),
        ("source_url_or_record_identifier", pa.string()),
        ("retrieval_timestamp", pa.string()),
        ("source_snapshot_version", pa.string()),
        ("derivation_method", pa.string()),
        ("derivation_rule_version", pa.string()),
        ("confidence_class", pa.string()),
        ("validation_status", pa.string()),
        ("conflict_status", pa.string()),
        ("reviewer_status", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class DerivedAssertion:
    assertion_id: str
    source_row_id: str
    gbif_id: str
    media_assertion_id: str | None
    target_field: str
    original_value: str | None
    derived_value: str | None
    evidence_source: str
    source_url_or_record_identifier: str | None
    retrieval_timestamp: str
    source_snapshot_version: str
    derivation_method: str
    derivation_rule_version: str
    confidence_class: str
    validation_status: str
    conflict_status: str
    reviewer_status: str | None

    def to_row(self) -> dict[str, object]:
        return {
            "assertion_version": DERIVED_ASSERTION_VERSION,
            "assertion_id": self.assertion_id,
            "source_row_id": self.source_row_id,
            "gbifID": self.gbif_id,
            "media_assertion_id": self.media_assertion_id,
            "target_field": self.target_field,
            "original_value": self.original_value,
            "derived_value": self.derived_value,
            "evidence_source": self.evidence_source,
            "source_url_or_record_identifier": self.source_url_or_record_identifier,
            "retrieval_timestamp": self.retrieval_timestamp,
            "source_snapshot_version": self.source_snapshot_version,
            "derivation_method": self.derivation_method,
            "derivation_rule_version": self.derivation_rule_version,
            "confidence_class": self.confidence_class,
            "validation_status": self.validation_status,
            "conflict_status": self.conflict_status,
            "reviewer_status": self.reviewer_status,
        }


def build_assertion(
    *,
    source_snapshot_version: str,
    source_row_id: str,
    gbif_id: str,
    target_field: str,
    original_value: object | None,
    derived_value: object | None,
    evidence_source: str,
    derivation_method: str,
    derivation_rule_version: str,
    confidence_class: str,
    validation_status: str,
    conflict_status: str,
    retrieval_timestamp: str,
    source_url_or_record_identifier: str | None = None,
    media_assertion_id: str | None = None,
    reviewer_status: str | None = None,
) -> DerivedAssertion:
    if confidence_class not in CONFIDENCE_CLASSES:
        raise ValueError(f"invalid confidence class: {confidence_class}")
    required = {
        "source_snapshot_version": source_snapshot_version,
        "source_row_id": source_row_id,
        "gbif_id": gbif_id,
        "target_field": target_field,
        "evidence_source": evidence_source,
        "derivation_method": derivation_method,
        "derivation_rule_version": derivation_rule_version,
        "validation_status": validation_status,
        "conflict_status": conflict_status,
        "retrieval_timestamp": retrieval_timestamp,
    }
    blank = [name for name, value in required.items() if not str(value or "").strip()]
    if blank:
        raise ValueError(f"derived assertion has blank required fields: {blank}")
    original = None if original_value is None else str(original_value)
    derived = None if derived_value is None else str(derived_value)
    identity = "|".join(
        (
            DERIVED_ASSERTION_VERSION,
            source_snapshot_version,
            source_row_id,
            media_assertion_id or "",
            target_field,
            original or "",
            derived or "",
            derivation_method,
            derivation_rule_version,
        )
    )
    return DerivedAssertion(
        assertion_id="sha256:" + hashlib.sha256(identity.encode()).hexdigest(),
        source_row_id=source_row_id,
        gbif_id=gbif_id,
        media_assertion_id=media_assertion_id,
        target_field=target_field,
        original_value=original,
        derived_value=derived,
        evidence_source=evidence_source,
        source_url_or_record_identifier=source_url_or_record_identifier,
        retrieval_timestamp=retrieval_timestamp,
        source_snapshot_version=source_snapshot_version,
        derivation_method=derivation_method,
        derivation_rule_version=derivation_rule_version,
        confidence_class=confidence_class,
        validation_status=validation_status,
        conflict_status=conflict_status,
        reviewer_status=reviewer_status,
    )


def assertion_table(assertions: list[DerivedAssertion]) -> pa.Table:
    rows = [assertion.to_row() for assertion in assertions]
    if len({row["assertion_id"] for row in rows}) != len(rows):
        raise ValueError("derived assertion identifiers must be unique")
    return pa.Table.from_pylist(rows, schema=DERIVED_ASSERTION_SCHEMA)


__all__ = [
    "CONFIDENCE_CLASSES",
    "DERIVED_ASSERTION_SCHEMA",
    "DERIVED_ASSERTION_VERSION",
    "DerivedAssertion",
    "assertion_table",
    "build_assertion",
]
