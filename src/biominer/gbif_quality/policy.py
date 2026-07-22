from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pyarrow as pa


FIELD_POLICY_SCHEMA_VERSION = "biominer-gbif-media-field-policy/v1"
FIELD_POLICY_SCHEMA = pa.schema(
    [
        ("policy_version", pa.string()),
        ("field_name", pa.string()),
        ("scope", pa.string()),
        ("required_status", pa.string()),
        ("applicability_rule", pa.string()),
        ("source_physical_type", pa.string()),
        ("valid_type", pa.string()),
        ("valid_vocabulary", pa.list_(pa.string())),
        ("allowed_derivation_sources", pa.list_(pa.string())),
        ("automatic_repair_policy", pa.string()),
        ("downstream_importance", pa.string()),
        ("quality_dimension", pa.string()),
        ("preserve_original", pa.bool_()),
        ("policy_note", pa.string()),
    ]
)

MEDIA_FIELDS = frozenset(
    {
        "media_type",
        "media_format",
        "media_identifier",
        "media_references",
        "media_created",
        "media_creator",
        "media_publisher",
        "media_license",
        "media_rightsHolder",
    }
)
INTEGER_FIELDS = frozenset(
    {
        "individualCount",
        "startDayOfYear",
        "endDayOfYear",
        "year",
        "month",
        "day",
        "taxonKey",
        "acceptedTaxonKey",
        "kingdomKey",
        "phylumKey",
        "classKey",
        "orderKey",
        "superfamilyKey",
        "familyKey",
        "subfamilyKey",
        "tribeKey",
        "genusKey",
        "subgenusKey",
        "speciesKey",
    }
)
FLOAT_FIELDS = frozenset(
    {"decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters"}
)
BOOLEAN_FIELDS = frozenset(
    {"hasCoordinate", "hasGeospatialIssues", "repatriated", "isSequenced"}
)
DATE_FIELDS = frozenset({"eventDate", "dateIdentified", "media_created"})
TIMESTAMP_FIELDS = frozenset(
    {"modified", "lastInterpreted", "lastParsed", "lastCrawled"}
)
CONTROLLED_VOCABULARIES: dict[str, tuple[str, ...]] = {
    "basisOfRecord": (
        "HUMAN_OBSERVATION",
        "MACHINE_OBSERVATION",
        "PRESERVED_SPECIMEN",
        "MATERIAL_SAMPLE",
        "LIVING_SPECIMEN",
        "FOSSIL_SPECIMEN",
        "OCCURRENCE",
    ),
    "occurrenceStatus": ("PRESENT", "ABSENT"),
    "sex": ("female", "male", "hermaphrodite"),
    "taxonomicStatus": (
        "ACCEPTED",
        "DOUBTFUL",
        "SYNONYM",
        "HETEROTYPIC_SYNONYM",
        "HOMOTYPIC_SYNONYM",
        "PROPARTE_SYNONYM",
    ),
    "media_type": ("StillImage",),
}


@dataclass(frozen=True, slots=True)
class FieldPolicy:
    field_name: str
    scope: str
    required_status: str
    applicability_rule: str
    source_physical_type: str
    valid_type: str
    valid_vocabulary: tuple[str, ...]
    allowed_derivation_sources: tuple[str, ...]
    automatic_repair_policy: str
    downstream_importance: str
    quality_dimension: str
    preserve_original: bool
    policy_note: str

    def to_row(self) -> dict[str, object]:
        return {
            "policy_version": FIELD_POLICY_SCHEMA_VERSION,
            "field_name": self.field_name,
            "scope": self.scope,
            "required_status": self.required_status,
            "applicability_rule": self.applicability_rule,
            "source_physical_type": self.source_physical_type,
            "valid_type": self.valid_type,
            "valid_vocabulary": list(self.valid_vocabulary),
            "allowed_derivation_sources": list(self.allowed_derivation_sources),
            "automatic_repair_policy": self.automatic_repair_policy,
            "downstream_importance": self.downstream_importance,
            "quality_dimension": self.quality_dimension,
            "preserve_original": self.preserve_original,
            "policy_note": self.policy_note,
        }


def build_field_policy(schema: pa.Schema) -> tuple[FieldPolicy, ...]:
    """Create the complete, conservative v1 policy for a physical schema."""

    policies = tuple(_policy(field) for field in schema)
    names = tuple(policy.field_name for policy in policies)
    if names != tuple(schema.names) or len(set(names)) != len(names):
        raise ValueError("field policy must cover the schema exactly once and in order")
    return policies


def field_policy_table(policies: Iterable[FieldPolicy]) -> pa.Table:
    return pa.Table.from_pylist(
        [policy.to_row() for policy in policies], schema=FIELD_POLICY_SCHEMA
    )


def _policy(field: pa.Field) -> FieldPolicy:
    name = field.name
    scope = "media_assertion" if name in MEDIA_FIELDS else "occurrence"
    required, rule, note = _applicability(name, scope)
    derivation, repair = _derivation(name)
    return FieldPolicy(
        field_name=name,
        scope=scope,
        required_status=required,
        applicability_rule=rule,
        source_physical_type=str(field.type),
        valid_type=_valid_type(name),
        valid_vocabulary=CONTROLLED_VOCABULARIES.get(name, ()),
        allowed_derivation_sources=derivation,
        automatic_repair_policy=repair,
        downstream_importance=_importance(name),
        quality_dimension=_quality_dimension(name),
        preserve_original=True,
        policy_note=note,
    )


def _applicability(name: str, scope: str) -> tuple[str, str, str]:
    overrides = {
        "gbifID": (
            "universally_required",
            "all_rows",
            "Required source occurrence key; it is not assumed permanent.",
        ),
        "species": (
            "required_for_specific_taxon_rank",
            "species_rank_or_below",
            "A valid genus-rank record is not a species completeness failure.",
        ),
        "specificEpithet": (
            "required_for_specific_taxon_rank",
            "species_rank_or_below",
            "Applicable only to species-rank or infraspecific records.",
        ),
        "infraspecificEpithet": (
            "required_for_specific_taxon_rank",
            "below_species_rank",
            "Applicable only below species rank.",
        ),
        "coordinateUncertaintyInMeters": (
            "conditionally_desirable",
            "coordinates_present",
            "Unknown uncertainty must remain unknown and is never written as zero.",
        ),
        "year": (
            "conditionally_desirable",
            "event_date_has_year_precision",
            "Original temporal fields remain unchanged.",
        ),
        "month": (
            "conditionally_desirable",
            "event_date_has_month_precision",
            "Month is not applicable to deliberately year-precision dates.",
        ),
        "day": (
            "conditionally_desirable",
            "event_date_has_day_precision",
            "Day is not applicable to deliberately month- or year-precision dates.",
        ),
        "media_format": (
            "required_for_specific_media_type",
            "direct_media_resource",
            "Format is applicable only when a direct media identifier exists.",
        ),
        "media_license": (
            "required_for_specific_media_type",
            "media_assertion",
            "Media licence is distinct from occurrence licence.",
        ),
        "identifiedBy": (
            "conditionally_desirable",
            "occurrence",
            "Must not be inferred from taxonomic acceptance.",
        ),
        "identificationVerificationStatus": (
            "conditionally_desirable",
            "occurrence",
            "Must not be inferred from identifiedBy; v3 contains a legacy transformation.",
        ),
        "individualCount": (
            "optional",
            "occurrence",
            "Not universally required; assess with occurrenceStatus.",
        ),
        "locationID": ("optional", "occurrence", "Not universally required."),
        "locality": (
            "potentially_withheld",
            "occurrence",
            "Absence may reflect withholding or generalization.",
        ),
        "media_creator": (
            "conditionally_desirable",
            "media_assertion",
            "Creator and rights holder are separate assertions.",
        ),
        "media_rightsHolder": (
            "conditionally_desirable",
            "media_assertion",
            "Creator and rights holder are separate assertions.",
        ),
    }
    if name in overrides:
        return overrides[name]
    if scope == "media_assertion":
        return "optional", "media_assertion", "Media assertion field."
    return "optional", "occurrence", "Optional occurrence field."


def _valid_type(name: str) -> str:
    if name in INTEGER_FIELDS:
        return "nullable_integer"
    if name in FLOAT_FIELDS:
        return "nullable_float"
    if name in BOOLEAN_FIELDS:
        return "nullable_boolean"
    if name in TIMESTAMP_FIELDS:
        return "timestamp"
    if name in DATE_FIELDS:
        return "date_or_interval"
    if name in CONTROLLED_VOCABULARIES:
        return "controlled_vocabulary"
    if name in {"issue", "nonTaxonomicIssue"}:
        return "multi_value_vocabulary"
    if name == "dynamicProperties":
        return "json_or_preserved_text"
    return "string"


def _derivation(name: str) -> tuple[tuple[str, ...], str]:
    if name in {"year", "month", "day"}:
        return ("eventDate",), "derive_to_separate_assertion_only"
    if name == "media_format":
        return ("verified_http_content_type",), "derive_to_separate_assertion_only"
    return (), "never_overwrite_original"


def _importance(name: str) -> str:
    if name in {
        "gbifID",
        "datasetKey",
        "occurrenceID",
        "species",
        "taxonRank",
        "media_identifier",
        "media_license",
    }:
        return "critical"
    if name in {
        "decimalLatitude",
        "decimalLongitude",
        "eventDate",
        "identifiedBy",
        "media_creator",
        "media_format",
    }:
        return "high"
    return "standard"


def _quality_dimension(name: str) -> str:
    if name.startswith("media_"):
        return "media"
    if name in {"year", "month", "day", "eventDate", "eventTime"}:
        return "temporal"
    if name in {
        "decimalLatitude",
        "decimalLongitude",
        "coordinateUncertaintyInMeters",
        "countryCode",
        "locality",
        "stateProvince",
    }:
        return "geospatial"
    if name in {
        "taxonID",
        "scientificName",
        "acceptedScientificName",
        "taxonRank",
        "species",
        "genus",
        "family",
        "order",
    } or name.endswith("TaxonKey") or name.endswith("Key"):
        return "taxonomic"
    if name in {
        "identifiedBy",
        "identificationID",
        "identificationVerificationStatus",
        "dateIdentified",
    }:
        return "identification"
    if name in {"license", "rightsHolder"}:
        return "occurrence_rights_context"
    return "occurrence_metadata"


__all__ = [
    "FIELD_POLICY_SCHEMA",
    "FIELD_POLICY_SCHEMA_VERSION",
    "FieldPolicy",
    "build_field_policy",
    "field_policy_table",
]
