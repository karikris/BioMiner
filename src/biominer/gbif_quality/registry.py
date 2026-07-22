from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

import pyarrow as pa

from biominer.common.semantic_hash import canonical_semantic_fingerprint


CHECK_REGISTRY_VERSION = "biominer-gbif-media-check-registry/v1"


class QualityStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    WITHHELD = "WITHHELD"
    GENERALIZED = "GENERALIZED"
    CONFLICT = "CONFLICT"
    NOT_TESTED = "NOT_TESTED"


CHECK_REGISTRY_SCHEMA = pa.schema(
    [
        ("registry_version", pa.string()),
        ("registry_fingerprint", pa.string()),
        ("check_id", pa.string()),
        ("check_family", pa.string()),
        ("scope", pa.string()),
        ("fields_inspected", pa.list_(pa.string())),
        ("applicability_rule", pa.string()),
        ("result_type", pa.string()),
        ("severity", pa.string()),
        ("deterministic", pa.bool_()),
        ("automatic_repair_permitted", pa.bool_()),
        ("required_evidence", pa.list_(pa.string())),
        ("repair_method", pa.string()),
        ("output_field_or_flag", pa.string()),
        ("rule_version", pa.string()),
        ("network_required", pa.bool_()),
        ("description", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class CheckDefinition:
    check_id: str
    check_family: str
    scope: str
    fields_inspected: tuple[str, ...]
    applicability_rule: str
    result_type: str
    severity: str
    deterministic: bool
    automatic_repair_permitted: bool
    required_evidence: tuple[str, ...]
    repair_method: str | None
    output_field_or_flag: str
    rule_version: str
    network_required: bool
    description: str

    def semantic_row(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "check_family": self.check_family,
            "scope": self.scope,
            "fields_inspected": list(self.fields_inspected),
            "applicability_rule": self.applicability_rule,
            "result_type": self.result_type,
            "severity": self.severity,
            "deterministic": self.deterministic,
            "automatic_repair_permitted": self.automatic_repair_permitted,
            "required_evidence": list(self.required_evidence),
            "repair_method": self.repair_method,
            "output_field_or_flag": self.output_field_or_flag,
            "rule_version": self.rule_version,
            "network_required": self.network_required,
            "description": self.description,
        }


def check_registry() -> tuple[CheckDefinition, ...]:
    """Return the immutable local/network check catalogue for v4."""

    checks = (
        _check("SOURCE_001", "source", "source", (), "source_artifact", "structural", "error", "source_checksum_status", "Pinned checksums and row counts match."),
        _check("SCHEMA_001", "schema", "source", (), "parquet_artifact", "structural", "error", "parquet_integrity_status", "Parquet metadata, row groups, and chunk bounds reconcile."),
        _check("SCHEMA_002", "schema", "source", (), "parquet_column", "categorical", "warning", "type_drift_status", "A physical column has one consistent type across parts and row groups."),
        _check("ID_001", "identifier", "occurrence", ("gbifID",), "all_occurrences", "categorical", "error", "gbif_id_status", "gbifID is present and contains ASCII digits only."),
        _check("ID_002", "identifier", "occurrence", ("datasetKey",), "all_occurrences", "categorical", "error", "dataset_key_status", "datasetKey is a canonical UUID."),
        _check("ID_003", "identifier", "occurrence", ("occurrenceID",), "all_occurrences", "categorical", "warning", "occurrence_id_status", "occurrenceID is present; URI syntax is recorded separately."),
        _check("ID_004", "identifier", "occurrence", ("datasetKey", "occurrenceID", "gbifID"), "dataset_and_occurrence_id_present", "categorical", "error", "occurrence_identity_conflict_status", "One datasetKey and occurrenceID pair does not map to contradictory gbifIDs."),
        _check("ID_005", "identifier", "media_assertion", ("gbifID", "media_identifier", "media_references"), "all_media_assertions", "hash", "error", "media_assertion_id", "A stable source-bound media assertion identifier is assigned."),
        _check("NULL_001", "semantic_null", "occurrence", (), "field_policy", "categorical", "warning", "semantic_null_status", "Null, blank, sentinel, withheld, generalized, and not-applicable states remain distinct."),
        _check("VOCAB_001", "vocabulary", "occurrence", ("basisOfRecord",), "value_present", "categorical", "warning", "basis_of_record_status", "basisOfRecord is in the pinned vocabulary."),
        _check("VOCAB_002", "vocabulary", "occurrence", ("occurrenceStatus",), "value_present", "categorical", "warning", "occurrence_status_vocabulary_status", "occurrenceStatus is PRESENT or ABSENT."),
        _check("VOCAB_003", "vocabulary", "occurrence", ("sex",), "value_present", "categorical", "info", "sex_vocabulary_status", "sex is a recognized Darwin Core term without overwriting the source value."),
        _check("VOCAB_004", "gbif_issue", "occurrence", ("issue",), "all_occurrences", "list", "info", "gbif_issue_flags", "GBIF issue strings are split into retained individual flags."),
        _check("TIME_001", "temporal", "occurrence", ("eventDate",), "event_date_present", "categorical", "warning", "event_date_status", "eventDate has supported syntax and valid calendar components."),
        _check("TIME_002", "temporal", "occurrence", ("eventDate", "year", "month", "day"), "event_date_and_component_present", "categorical", "warning", "temporal_component_conflict_status", "Explicit temporal components agree with eventDate precision."),
        _check("GEO_001", "geospatial", "occurrence", ("decimalLatitude", "decimalLongitude"), "either_coordinate_present", "categorical", "error", "coordinate_pair_status", "Coordinates form a complete numeric pair within WGS84 ranges."),
        _check("GEO_002", "geospatial", "occurrence", ("decimalLatitude", "decimalLongitude"), "coordinate_pair_present", "categorical", "warning", "zero_coordinate_status", "The coordinate pair is not the 0,0 sentinel."),
        _check("GEO_003", "geospatial", "occurrence", ("coordinateUncertaintyInMeters", "decimalLatitude", "decimalLongitude"), "coordinate_pair_present", "categorical", "warning", "coordinate_uncertainty_status", "Coordinate uncertainty is absent/unknown or finite and non-negative."),
        _check("TAXON_001", "taxonomic", "occurrence", ("taxonRank", "species", "specificEpithet"), "rank_present", "categorical", "error", "rank_name_consistency_status", "Species-rank records carry species-level source or interpreted evidence."),
        _check("TAXON_002", "taxonomic", "occurrence", ("taxonKey", "acceptedTaxonKey", "taxonomicStatus"), "taxon_key_present", "categorical", "warning", "accepted_taxon_key_status", "Accepted taxon key is present where the interpreted match supports it."),
        _check("TAXON_003", "taxonomic", "occurrence", ("kingdom", "phylum", "class", "order", "family", "genus", "species"), "taxonomic_interpretation_present", "categorical", "error", "taxonomic_hierarchy_status", "The interpreted hierarchy is internally consistent with the pinned local snapshot."),
        _check("IDENT_001", "identification", "occurrence", ("identifiedBy",), "all_occurrences", "categorical", "info", "identified_by_status", "identifiedBy presence is reported without synthesizing provenance."),
        _check("IDENT_002", "identification", "occurrence", ("identificationVerificationStatus",), "all_occurrences", "categorical", "warning", "verification_source_evidence_status", "Legacy transformed verification text is not treated as publisher source evidence."),
        _check("OCCURRENCE_001", "occurrence_semantics", "occurrence", ("occurrenceStatus", "individualCount"), "either_value_present", "categorical", "warning", "occurrence_count_consistency_status", "ABSENT does not conflict with a positive individualCount."),
        _check("MEDIA_URL_001", "media_url", "media_assertion", ("media_identifier",), "media_identifier_present", "categorical", "error", "direct_media_url_status", "Direct media identifier uses HTTP or HTTPS syntax."),
        _check("MEDIA_URL_002", "media_url", "media_assertion", ("media_references",), "media_reference_present", "categorical", "warning", "media_reference_url_status", "Media reference uses HTTP or HTTPS syntax."),
        _check("MEDIA_FILE_001", "media_file", "media_assertion", ("media_type", "media_format"), "direct_media_resource", "categorical", "warning", "media_type_format_status", "Declared media type and MIME format are syntactically compatible."),
        _check("RIGHTS_001", "rights", "media_assertion", ("media_license", "license"), "all_media_assertions", "categorical", "error", "media_rights_status", "Media licence is assessed independently from occurrence licence."),
        _check("PROVENANCE_001", "provenance", "media_assertion", ("identificationVerificationStatus",), "all_media_assertions", "categorical", "error", "legacy_transformation_provenance_status", "Known v3 legacy transformations are explicitly labelled."),
        _check("MEDIA_URL_003", "media_url", "media_resource", ("media_identifier",), "network_eligible_url", "categorical", "warning", "network_url_status", "Live URL status is explicit and never inferred from syntax.", network_required=True, deterministic=False),
    )
    _validate_registry(checks)
    return checks


def registry_fingerprint(checks: Iterable[CheckDefinition] | None = None) -> str:
    rows = tuple(checks or check_registry())
    return canonical_semantic_fingerprint(
        {
            "registry_version": CHECK_REGISTRY_VERSION,
            "checks": [check.semantic_row() for check in rows],
        }
    )


def check_registry_table(
    checks: Iterable[CheckDefinition] | None = None,
) -> pa.Table:
    rows = tuple(checks or check_registry())
    _validate_registry(rows)
    fingerprint = registry_fingerprint(rows)
    return pa.Table.from_pylist(
        [
            {
                "registry_version": CHECK_REGISTRY_VERSION,
                "registry_fingerprint": fingerprint,
                **check.semantic_row(),
            }
            for check in rows
        ],
        schema=CHECK_REGISTRY_SCHEMA,
    )


def _check(
    check_id: str,
    family: str,
    scope: str,
    fields: tuple[str, ...],
    applicability: str,
    result_type: str,
    severity: str,
    output: str,
    description: str,
    *,
    deterministic: bool = True,
    network_required: bool = False,
) -> CheckDefinition:
    return CheckDefinition(
        check_id=check_id,
        check_family=family,
        scope=scope,
        fields_inspected=fields,
        applicability_rule=applicability,
        result_type=result_type,
        severity=severity,
        deterministic=deterministic,
        automatic_repair_permitted=False,
        required_evidence=fields or ("artifact_manifest",),
        repair_method=None,
        output_field_or_flag=output,
        rule_version="1.0.0",
        network_required=network_required,
        description=description,
    )


def _validate_registry(checks: tuple[CheckDefinition, ...]) -> None:
    identifiers = [check.check_id for check in checks]
    if not checks or len(identifiers) != len(set(identifiers)):
        raise ValueError("check registry identifiers must be nonempty and unique")
    allowed_scopes = {
        "source",
        "occurrence",
        "media_assertion",
        "media_resource",
        "provider",
        "dataset",
        "aggregate",
    }
    allowed_severities = {"info", "warning", "error"}
    for check in checks:
        if check.scope not in allowed_scopes:
            raise ValueError(f"invalid check scope: {check.check_id}")
        if check.severity not in allowed_severities:
            raise ValueError(f"invalid check severity: {check.check_id}")
        if check.network_required and check.deterministic:
            raise ValueError(f"network check cannot claim deterministic: {check.check_id}")
        if check.automatic_repair_permitted and not check.repair_method:
            raise ValueError(f"repairable check lacks repair method: {check.check_id}")


__all__ = [
    "CHECK_REGISTRY_SCHEMA",
    "CHECK_REGISTRY_VERSION",
    "CheckDefinition",
    "QualityStatus",
    "check_registry",
    "check_registry_table",
    "registry_fingerprint",
]
