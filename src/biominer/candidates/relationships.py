from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re

import polars as pl

from biominer.storage.parquet import write_parquet


COMPETITOR_RELATIONSHIPS_SCHEMA_VERSION = "competitor-relationships-v1.0.0"
COMPETITOR_RELATIONSHIP_SOURCE_SCHEMA_VERSION = (
    "competitor-relationship-source-v1.0.0"
)
COMPETITOR_RELATIONSHIPS_FILE = "competitor_relationships.parquet"

RELATIONSHIP_TYPES = frozenset(
    {
        "known_mimic",
        "close_congener",
        "historical_false_positive_species",
        "historical_false_positive_genus",
        "taxonomic_neighbour",
        "visual_neighbour",
    }
)
REVIEW_STATUSES = frozenset({"pending", "reviewed", "rejected", "superseded"})
OBJECT_SCOPE_TYPES = frozenset({"species", "genus"})

_PRIMARY_KEY = [
    "subject_accepted_taxon_key",
    "relationship_type",
    "object_scope_type",
    "object_scope_id",
    "evidence_version",
]
_LOGICAL_EDGE = _PRIMARY_KEY[:-1]
_SOURCE_ROW_FIELDS = frozenset(
    {
        "subject_accepted_taxon_key",
        "object_scope_type",
        "object_scope_id",
        "relationship_type",
        "source",
        "source_record_id",
        "evidence_version",
        "evidence_note",
        "review_status",
        "reviewed_by",
        "reviewed_at",
        "enabled",
        "prototype_fingerprint",
        "model_fingerprint",
    }
)
_SCOPE_BY_RELATIONSHIP = {
    "known_mimic": frozenset({"species"}),
    "close_congener": frozenset({"species"}),
    "historical_false_positive_species": frozenset({"species"}),
    "historical_false_positive_genus": frozenset({"genus"}),
    "taxonomic_neighbour": frozenset({"species", "genus"}),
    "visual_neighbour": frozenset({"species"}),
}
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


def competitor_relationships_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "subject_accepted_taxon_key": pl.String,
        "object_scope_type": pl.String,
        "object_scope_id": pl.String,
        "relationship_type": pl.String,
        "source": pl.String,
        "source_record_id": pl.String,
        "evidence_version": pl.String,
        "evidence_note": pl.String,
        "review_status": pl.String,
        "reviewed_by": pl.String,
        "reviewed_at": pl.String,
        "enabled": pl.Boolean,
        "prototype_fingerprint": pl.String,
        "model_fingerprint": pl.String,
        "relationship_fingerprint": pl.String,
    }


def load_competitor_relationship_source(
    source: str | Path | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(source, Mapping):
        payload = dict(source)
    elif isinstance(source, (str, Path)):
        path = Path(source)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid competitor relationship JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("competitor relationship source root must be an object")
    else:
        raise TypeError("source must be a path or mapping")

    expected_root_fields = {"schema_version", "relationships"}
    unknown = sorted(set(payload) - expected_root_fields)
    missing = sorted(expected_root_fields - set(payload))
    if missing:
        raise ValueError(f"competitor relationship source is missing fields: {missing}")
    if unknown:
        raise ValueError(f"competitor relationship source has unknown fields: {unknown}")
    if payload["schema_version"] != COMPETITOR_RELATIONSHIP_SOURCE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported competitor relationship source schema: "
            f"{payload['schema_version']!r}"
        )
    relationships = payload["relationships"]
    if not isinstance(relationships, list):
        raise TypeError("competitor relationship source relationships must be a list")
    return payload


def compile_competitor_relationships(
    source: str | Path | Mapping[str, object],
    taxa: pl.DataFrame,
) -> pl.DataFrame:
    if not isinstance(taxa, pl.DataFrame):
        raise TypeError("taxa must be a Polars DataFrame")
    payload = load_competitor_relationship_source(source)
    taxonomy = _accepted_taxonomy(taxa)
    rows: list[dict[str, object]] = []
    for index, value in enumerate(payload["relationships"]):
        if not isinstance(value, Mapping):
            raise TypeError(f"relationships[{index}] must be an object")
        rows.append(_compile_row(value, index=index, taxonomy=taxonomy))
    frame = _relationship_frame(rows)
    _validate_relationship_frame(frame, verify_fingerprints=True)
    return frame


def write_competitor_relationships(
    frame: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a Polars DataFrame")
    _validate_relationship_frame(frame, verify_fingerprints=True)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= COMPETITOR_RELATIONSHIPS_FILE
    return write_parquet(frame, destination, overwrite=overwrite)


def _compile_row(
    value: Mapping[str, object],
    *,
    index: int,
    taxonomy: dict[str, dict[str, str]],
) -> dict[str, object]:
    unknown = sorted(set(value) - _SOURCE_ROW_FIELDS)
    if unknown:
        raise ValueError(f"relationships[{index}] has unknown fields: {unknown}")
    missing = sorted(
        _SOURCE_ROW_FIELDS
        - {"prototype_fingerprint", "model_fingerprint"}
        - set(value)
    )
    if missing:
        raise ValueError(f"relationships[{index}] is missing fields: {missing}")

    subject_key = _required_text(
        value.get("subject_accepted_taxon_key"),
        field=f"relationships[{index}].subject_accepted_taxon_key",
    )
    subject = taxonomy.get(subject_key)
    if subject is None or subject["rank"] != "SPECIES":
        raise ValueError(f"relationship subject is not an accepted in-scope species: {subject_key}")

    relationship_type = _choice(
        value.get("relationship_type"),
        field=f"relationships[{index}].relationship_type",
        choices=RELATIONSHIP_TYPES,
    )
    scope_type = _choice(
        value.get("object_scope_type"),
        field=f"relationships[{index}].object_scope_type",
        choices=OBJECT_SCOPE_TYPES,
    )
    allowed_scopes = _SCOPE_BY_RELATIONSHIP[relationship_type]
    if scope_type not in allowed_scopes:
        raise ValueError(
            f"{relationship_type} requires object scope in {sorted(allowed_scopes)}, "
            f"not {scope_type!r}"
        )
    scope_id = _required_text(
        value.get("object_scope_id"),
        field=f"relationships[{index}].object_scope_id",
    )
    object_taxon = _resolve_object_taxon(
        scope_type=scope_type,
        scope_id=scope_id,
        taxonomy=taxonomy,
    )
    if scope_type == "species" and scope_id == subject_key:
        raise ValueError("competitor relationship cannot point a species to itself")
    if (
        relationship_type == "close_congener"
        and object_taxon["genus"] != subject["genus"]
    ):
        raise ValueError("close_congener object must share the subject genus")

    review_status = _choice(
        value.get("review_status"),
        field=f"relationships[{index}].review_status",
        choices=REVIEW_STATUSES,
    )
    enabled = value.get("enabled")
    if not isinstance(enabled, bool):
        raise TypeError(f"relationships[{index}].enabled must be boolean")
    reviewed_by = _optional_text(value.get("reviewed_by"))
    reviewed_at = _review_timestamp(value.get("reviewed_at"), index=index)
    if review_status == "pending":
        if reviewed_by is not None or reviewed_at is not None:
            raise ValueError("pending relationship cannot have review provenance")
    elif reviewed_by is None or reviewed_at is None:
        raise ValueError(f"{review_status} relationship requires review provenance")
    if enabled and review_status != "reviewed":
        raise ValueError("enabled competitor relationship must be reviewed")

    prototype_fingerprint = _optional_fingerprint(
        value.get("prototype_fingerprint"),
        field=f"relationships[{index}].prototype_fingerprint",
    )
    model_fingerprint = _optional_fingerprint(
        value.get("model_fingerprint"),
        field=f"relationships[{index}].model_fingerprint",
    )
    if relationship_type == "visual_neighbour":
        if prototype_fingerprint is None or model_fingerprint is None:
            raise ValueError(
                "visual_neighbour relationship requires prototype and model fingerprints"
            )
    elif prototype_fingerprint is not None or model_fingerprint is not None:
        raise ValueError(
            "prototype and model fingerprints are reserved for visual_neighbour relationships"
        )

    row: dict[str, object] = {
        "schema_version": COMPETITOR_RELATIONSHIPS_SCHEMA_VERSION,
        "subject_accepted_taxon_key": subject_key,
        "object_scope_type": scope_type,
        "object_scope_id": scope_id,
        "relationship_type": relationship_type,
        "source": _required_text(
            value.get("source"), field=f"relationships[{index}].source"
        ),
        "source_record_id": _required_text(
            value.get("source_record_id"),
            field=f"relationships[{index}].source_record_id",
        ),
        "evidence_version": _required_text(
            value.get("evidence_version"),
            field=f"relationships[{index}].evidence_version",
        ),
        "evidence_note": _required_text(
            value.get("evidence_note"),
            field=f"relationships[{index}].evidence_note",
        ),
        "review_status": review_status,
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
        "enabled": enabled,
        "prototype_fingerprint": prototype_fingerprint,
        "model_fingerprint": model_fingerprint,
    }
    row["relationship_fingerprint"] = _relationship_fingerprint(row)
    return row


def _accepted_taxonomy(taxa: pl.DataFrame) -> dict[str, dict[str, str]]:
    required = {
        "accepted_taxon_key",
        "scientific_name",
        "rank",
        "taxonomic_status",
        "family",
        "genus",
        "in_scope",
    }
    missing = sorted(required - set(taxa.columns))
    if missing:
        raise ValueError(f"taxa is missing required columns: {missing}")
    taxonomy: dict[str, dict[str, str]] = {}
    accepted = taxa.filter(
        (pl.col("taxonomic_status").cast(pl.String).str.to_uppercase() == "ACCEPTED")
        & pl.col("in_scope").fill_null(False)
        & pl.col("rank").cast(pl.String).str.to_uppercase().is_in(["SPECIES", "GENUS"])
    )
    for row in accepted.iter_rows(named=True):
        key = _required_text(row.get("accepted_taxon_key"), field="accepted_taxon_key")
        rank = _required_text(row.get("rank"), field="rank").upper()
        scientific_name = _required_text(
            row.get("scientific_name"), field="scientific_name"
        )
        taxon = {
            "accepted_taxon_key": key,
            "scientific_name": scientific_name,
            "rank": rank,
            "family": _required_text(row.get("family"), field="family"),
            "genus": _required_text(row.get("genus"), field="genus"),
        }
        existing = taxonomy.get(key)
        if existing is not None and existing != taxon:
            raise ValueError(f"conflicting accepted taxonomy rows for {key}")
        taxonomy[key] = taxon
    return taxonomy


def _resolve_object_taxon(
    *,
    scope_type: str,
    scope_id: str,
    taxonomy: dict[str, dict[str, str]],
) -> dict[str, str]:
    if scope_type == "species":
        taxon = taxonomy.get(scope_id)
        if taxon is None or taxon["rank"] != "SPECIES":
            raise ValueError(
                f"relationship object is not an accepted in-scope species: {scope_id}"
            )
        return taxon
    matches = [
        taxon
        for taxon in taxonomy.values()
        if taxon["rank"] == "GENUS" and taxon["scientific_name"] == scope_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"relationship object is not one accepted in-scope genus: {scope_id}"
        )
    return matches[0]


def _validate_relationship_frame(
    frame: pl.DataFrame,
    *,
    verify_fingerprints: bool,
) -> None:
    schema = competitor_relationships_schema()
    if frame.schema != schema:
        raise ValueError("competitor relationship frame does not match the physical schema")
    expected = frame.sort(_PRIMARY_KEY)
    if not frame.equals(expected):
        raise ValueError("competitor relationship frame is not in deterministic sort order")
    duplicate_primary = frame.group_by(_PRIMARY_KEY).len().filter(pl.col("len") > 1)
    if not duplicate_primary.is_empty():
        raise ValueError("competitor relationship frame contains duplicate primary keys")
    duplicate_sources = (
        frame.group_by(["source", "source_record_id"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicate_sources.is_empty():
        raise ValueError("competitor relationship source record IDs are not unique")
    duplicate_enabled = (
        frame.filter(pl.col("enabled"))
        .group_by(_LOGICAL_EDGE)
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicate_enabled.is_empty():
        raise ValueError("competitor relationship has multiple enabled evidence versions")
    if verify_fingerprints:
        for row in frame.iter_rows(named=True):
            expected_fingerprint = _relationship_fingerprint(
                {key: value for key, value in row.items() if key != "relationship_fingerprint"}
            )
            if row["relationship_fingerprint"] != expected_fingerprint:
                raise ValueError("competitor relationship fingerprint mismatch")


def _relationship_frame(rows: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    schema = competitor_relationships_schema()
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, strict=True).sort(_PRIMARY_KEY)


def _relationship_fingerprint(row: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(row),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _review_timestamp(value: object, *, index: int) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"relationships[{index}].reviewed_at must be ISO 8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"relationships[{index}].reviewed_at must include a timezone")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _choice(value: object, *, field: str, choices: frozenset[str]) -> str:
    text = _required_text(value, field=field)
    if text not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}")
    return text


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_fingerprint(value: object, *, field: str) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 digest")
    return text


__all__ = [
    "COMPETITOR_RELATIONSHIPS_FILE",
    "COMPETITOR_RELATIONSHIPS_SCHEMA_VERSION",
    "COMPETITOR_RELATIONSHIP_SOURCE_SCHEMA_VERSION",
    "OBJECT_SCOPE_TYPES",
    "RELATIONSHIP_TYPES",
    "REVIEW_STATUSES",
    "compile_competitor_relationships",
    "competitor_relationships_schema",
    "load_competitor_relationship_source",
    "write_competitor_relationships",
]
