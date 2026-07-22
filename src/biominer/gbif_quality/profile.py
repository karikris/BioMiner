from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.policy import FieldPolicy


COMPLETENESS_SCHEMA_VERSION = "biominer-gbif-media-completeness/v1"
COMPLETENESS_SCHEMA = pa.schema(
    [
        ("profile_version", pa.string()),
        ("field_index", pa.int32()),
        ("field_name", pa.string()),
        ("scope", pa.string()),
        ("required_status", pa.string()),
        ("applicability_rule", pa.string()),
        ("media_row_denominator", pa.int64()),
        ("physical_filled_media_rows", pa.int64()),
        ("physical_missing_media_rows", pa.int64()),
        ("physical_fill_media_pct", pa.float64()),
        ("semantic_sentinel_media_rows", pa.int64()),
        ("literal_zero_media_rows", pa.int64()),
        ("applicable_media_rows", pa.int64()),
        ("applicable_filled_media_rows", pa.int64()),
        ("applicable_fill_media_pct", pa.float64()),
        ("not_applicable_media_rows", pa.int64()),
        ("withheld_media_rows", pa.int64()),
        ("generalized_media_rows", pa.int64()),
        ("non_repairable_null_media_rows", pa.int64()),
        ("occurrence_denominator", pa.int64()),
        ("physical_filled_occurrences", pa.int64()),
        ("physical_missing_occurrences", pa.int64()),
        ("physical_fill_occurrence_pct", pa.float64()),
        ("semantic_sentinel_occurrences", pa.int64()),
        ("literal_zero_occurrences", pa.int64()),
        ("applicable_occurrences", pa.int64()),
        ("applicable_filled_occurrences", pa.int64()),
        ("applicable_fill_occurrence_pct", pa.float64()),
        ("not_applicable_occurrences", pa.int64()),
        ("withheld_occurrences", pa.int64()),
        ("generalized_occurrences", pa.int64()),
        ("non_repairable_null_occurrences", pa.int64()),
        ("structurally_absent_media_rows", pa.int64()),
        ("structurally_absent_occurrences", pa.int64()),
        ("repairable_null_media_rows", pa.int64()),
        ("repairable_null_occurrences", pa.int64()),
        ("invalid_present_media_rows", pa.int64()),
        ("invalid_present_occurrences", pa.int64()),
        ("conflict_media_rows", pa.int64()),
        ("conflict_occurrences", pa.int64()),
        ("validity_check_status", pa.string()),
        ("repairability_status", pa.string()),
        ("conflict_check_status", pa.string()),
    ]
)

_SEMANTIC_SENTINELS = (
    "na",
    "n/a",
    "null",
    "none",
    "unknown",
    "not recorded",
)
_SPECIES_RANKS = (
    "species",
    "subspecies",
    "variety",
    "form",
    "infraspecific name",
)
_BELOW_SPECIES_RANKS = (
    "subspecies",
    "variety",
    "form",
    "infraspecific name",
)


@dataclass(frozen=True, slots=True)
class CompletenessProfile:
    schema_version: str
    rows: tuple[dict[str, Any], ...]
    denominators: dict[str, int]
    validation: dict[str, bool]

    def table(self) -> pa.Table:
        return pa.Table.from_pylist(list(self.rows), schema=COMPLETENESS_SCHEMA)

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "denominators": self.denominators,
            "validation": self.validation,
            "columns": list(self.rows),
        }


def profile_completeness(
    parquet_path: str | Path,
    policies: Iterable[FieldPolicy],
    *,
    occurrence_batch_size: int = 8,
    memory_limit: str = "4GB",
    temp_directory: str | Path | None = None,
) -> CompletenessProfile:
    """Profile exact row and occurrence completeness with bounded DuckDB scans."""

    path = Path(parquet_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    policy_rows = tuple(policies)
    schema = pq.ParquetFile(path).schema_arrow
    if tuple(policy.field_name for policy in policy_rows) != tuple(schema.names):
        raise ValueError("field policies do not exactly match the Parquet schema")
    if "gbifID" not in schema.names:
        raise ValueError("profile input must contain gbifID")
    if occurrence_batch_size < 1:
        raise ValueError("occurrence_batch_size must be positive")

    if temp_directory is None:
        with TemporaryDirectory(prefix="biominer-gbif-profile-") as temporary:
            return _profile(
                path,
                schema,
                policy_rows,
                occurrence_batch_size,
                memory_limit,
                Path(temporary),
            )
    temporary_path = Path(temp_directory).resolve()
    temporary_path.mkdir(parents=True, exist_ok=True)
    return _profile(
        path,
        schema,
        policy_rows,
        occurrence_batch_size,
        memory_limit,
        temporary_path,
    )


def _profile(
    path: Path,
    schema: pa.Schema,
    policies: tuple[FieldPolicy, ...],
    occurrence_batch_size: int,
    memory_limit: str,
    temporary: Path,
) -> CompletenessProfile:
    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit = {_literal(memory_limit)}")
        connection.execute(f"SET temp_directory = {_literal(str(temporary))}")
        connection.execute("SET preserve_insertion_order = false")
        relation = f"read_parquet({_literal(str(path))})"
        media = _media_counts(connection, relation, schema, policies)
        media_rows = int(media["row_count"])
        occurrence_rows = int(media["occurrence_count"])
        occurrences = _occurrence_counts(
            connection,
            relation,
            schema,
            policies,
            occurrence_batch_size,
        )
    finally:
        connection.close()

    rows = tuple(
        _profile_row(index, policy, media, occurrences, media_rows, occurrence_rows)
        for index, policy in enumerate(policies)
    )
    metadata_rows = pq.ParquetFile(path).metadata.num_rows
    validation = {
        "metadata_row_count_matches": media_rows == metadata_rows,
        "all_columns_profiled": len(rows) == len(schema),
        "physical_media_counts_reconcile": all(
            row["physical_filled_media_rows"]
            + row["physical_missing_media_rows"]
            == media_rows
            for row in rows
        ),
        "physical_occurrence_counts_reconcile": all(
            row["physical_filled_occurrences"]
            + row["physical_missing_occurrences"]
            == occurrence_rows
            for row in rows
        ),
        "applicable_media_counts_reconcile": all(
            row["applicable_media_rows"] + row["not_applicable_media_rows"]
            == media_rows
            for row in rows
        ),
        "applicable_occurrence_counts_reconcile": all(
            row["applicable_occurrences"] + row["not_applicable_occurrences"]
            == occurrence_rows
            for row in rows
        ),
        "semantic_media_nulls_partitioned": all(
            row["applicable_filled_media_rows"]
            + row["withheld_media_rows"]
            + row["generalized_media_rows"]
            + row["non_repairable_null_media_rows"]
            == row["applicable_media_rows"]
            for row in rows
        ),
        "semantic_occurrence_nulls_partitioned": all(
            row["applicable_filled_occurrences"]
            + row["withheld_occurrences"]
            + row["generalized_occurrences"]
            + row["non_repairable_null_occurrences"]
            == row["applicable_occurrences"]
            for row in rows
        ),
    }
    if not all(validation.values()):
        raise ValueError(f"completeness profile validation failed: {validation}")
    return CompletenessProfile(
        schema_version=COMPLETENESS_SCHEMA_VERSION,
        rows=rows,
        denominators={
            "media_rows": media_rows,
            "distinct_occurrences": occurrence_rows,
            "columns": len(schema),
        },
        validation=validation,
    )


def _media_counts(
    connection: duckdb.DuckDBPyConnection,
    relation: str,
    schema: pa.Schema,
    policies: tuple[FieldPolicy, ...],
) -> dict[str, int]:
    selections = [
        "count(*)::BIGINT AS row_count",
        (
            "count(DISTINCT CASE WHEN "
            f"{_meaningful('gbifID')} THEN {_identifier('gbifID')} END)"
            "::BIGINT AS occurrence_count"
        ),
    ]
    names = set(schema.names)
    for index, policy in enumerate(policies):
        expressions = _field_expressions(policy, names)
        for suffix, expression in expressions.items():
            selections.append(
                f"sum(CASE WHEN {expression} THEN 1 ELSE 0 END)::BIGINT "
                f"AS {_identifier(f'f{index}_{suffix}')}"
            )
    cursor = connection.execute(f"SELECT {', '.join(selections)} FROM {relation}")
    values = cursor.fetchone()
    assert values is not None
    return dict(zip((item[0] for item in cursor.description), map(int, values), strict=True))


def _occurrence_counts(
    connection: duckdb.DuckDBPyConnection,
    relation: str,
    schema: pa.Schema,
    policies: tuple[FieldPolicy, ...],
    batch_size: int,
) -> dict[str, int]:
    output: dict[str, int] = {}
    names = set(schema.names)
    for start in range(0, len(policies), batch_size):
        batch = policies[start : start + batch_size]
        per_occurrence = []
        aggregate = []
        for relative, policy in enumerate(batch):
            index = start + relative
            expressions = _field_expressions(policy, names)
            for suffix, expression in expressions.items():
                alias = f"f{index}_{suffix}"
                per_occurrence.append(
                    f"bool_or({expression}) AS {_identifier(alias)}"
                )
            for suffix in expressions:
                alias = f"f{index}_{suffix}"
                condition = _occurrence_aggregate_condition(index, suffix)
                aggregate.append(
                    f"sum(CASE WHEN {condition} THEN 1 ELSE 0 END)"
                    f"::BIGINT AS {_identifier(alias)}"
                )
        sql = (
            "WITH per_occurrence AS (SELECT gbifID, "
            + ", ".join(per_occurrence)
            + f" FROM {relation} WHERE {_meaningful('gbifID')} GROUP BY gbifID) "
            + "SELECT "
            + ", ".join(aggregate)
            + " FROM per_occurrence"
        )
        cursor = connection.execute(sql)
        values = cursor.fetchone()
        assert values is not None
        output.update(
            zip((item[0] for item in cursor.description), map(int, values), strict=True)
        )
    return output


def _occurrence_aggregate_condition(index: int, suffix: str) -> str:
    value = _identifier(f"f{index}_{suffix}")
    filled = _identifier(f"f{index}_applicable_filled")
    withheld = _identifier(f"f{index}_withheld")
    if suffix == "withheld":
        return f"({value}) AND NOT ({filled})"
    if suffix == "generalized":
        return f"({value}) AND NOT ({filled}) AND NOT ({withheld})"
    return value


def _field_expressions(
    policy: FieldPolicy, schema_names: set[str]
) -> dict[str, str]:
    field = policy.field_name
    physical = _nonblank(field)
    meaningful = _meaningful(field)
    applicable = _applicable(policy.applicability_rule, schema_names)
    withheld_context = (
        _meaningful("informationWithheld")
        if "informationWithheld" in schema_names
        else "false"
    )
    generalized_context = (
        _meaningful("dataGeneralizations")
        if "dataGeneralizations" in schema_names
        else "false"
    )
    missing_applicable = f"({applicable}) AND NOT ({meaningful})"
    withheld = f"({missing_applicable}) AND ({withheld_context})"
    generalized = (
        f"({missing_applicable}) AND NOT ({withheld_context}) "
        f"AND ({generalized_context})"
    )
    return {
        "physical": physical,
        "meaningful": meaningful,
        "sentinel": f"({physical}) AND NOT ({meaningful})",
        "zero": f"({physical}) AND trim(cast({_identifier(field)} AS VARCHAR)) = '0'",
        "applicable": applicable,
        "applicable_filled": f"({applicable}) AND ({meaningful})",
        "withheld": withheld,
        "generalized": generalized,
    }


def _profile_row(
    index: int,
    policy: FieldPolicy,
    media: dict[str, int],
    occurrences: dict[str, int],
    media_rows: int,
    occurrence_rows: int,
) -> dict[str, Any]:
    key = f"f{index}_"
    physical_media = media[key + "physical"]
    meaningful_media = media[key + "meaningful"]
    applicable_media = media[key + "applicable"]
    filled_media = media[key + "applicable_filled"]
    withheld_media = media[key + "withheld"]
    generalized_media = media[key + "generalized"]
    physical_occ = occurrences[key + "physical"]
    meaningful_occ = occurrences[key + "meaningful"]
    applicable_occ = occurrences[key + "applicable"]
    filled_occ = occurrences[key + "applicable_filled"]
    withheld_occ = occurrences[key + "withheld"]
    generalized_occ = occurrences[key + "generalized"]
    return {
        "profile_version": COMPLETENESS_SCHEMA_VERSION,
        "field_index": index,
        "field_name": policy.field_name,
        "scope": policy.scope,
        "required_status": policy.required_status,
        "applicability_rule": policy.applicability_rule,
        "media_row_denominator": media_rows,
        "physical_filled_media_rows": physical_media,
        "physical_missing_media_rows": media_rows - physical_media,
        "physical_fill_media_pct": _percentage(physical_media, media_rows),
        "semantic_sentinel_media_rows": media[key + "sentinel"],
        "literal_zero_media_rows": media[key + "zero"],
        "applicable_media_rows": applicable_media,
        "applicable_filled_media_rows": filled_media,
        "applicable_fill_media_pct": _percentage(filled_media, applicable_media),
        "not_applicable_media_rows": media_rows - applicable_media,
        "withheld_media_rows": withheld_media,
        "generalized_media_rows": generalized_media,
        "non_repairable_null_media_rows": (
            applicable_media - filled_media - withheld_media - generalized_media
        ),
        "occurrence_denominator": occurrence_rows,
        "physical_filled_occurrences": physical_occ,
        "physical_missing_occurrences": occurrence_rows - physical_occ,
        "physical_fill_occurrence_pct": _percentage(physical_occ, occurrence_rows),
        "semantic_sentinel_occurrences": occurrences[key + "sentinel"],
        "literal_zero_occurrences": occurrences[key + "zero"],
        "applicable_occurrences": applicable_occ,
        "applicable_filled_occurrences": filled_occ,
        "applicable_fill_occurrence_pct": _percentage(filled_occ, applicable_occ),
        "not_applicable_occurrences": occurrence_rows - applicable_occ,
        "withheld_occurrences": withheld_occ,
        "generalized_occurrences": generalized_occ,
        "non_repairable_null_occurrences": (
            applicable_occ - filled_occ - withheld_occ - generalized_occ
        ),
        "structurally_absent_media_rows": 0,
        "structurally_absent_occurrences": 0,
        "repairable_null_media_rows": 0,
        "repairable_null_occurrences": 0,
        "invalid_present_media_rows": None,
        "invalid_present_occurrences": None,
        "conflict_media_rows": None,
        "conflict_occurrences": None,
        "validity_check_status": "NOT_TESTED",
        "repairability_status": (
            "UNKNOWN" if applicable_media > filled_media else "NOT_APPLICABLE"
        ),
        "conflict_check_status": "NOT_TESTED",
    }


def _applicable(rule: str, names: set[str]) -> str:
    if rule in {"all_rows", "media_assertion", "occurrence"}:
        return "true"
    if rule == "species_rank_or_below":
        _require(names, "taxonRank", rule)
        return _in_values("taxonRank", _SPECIES_RANKS)
    if rule == "below_species_rank":
        _require(names, "taxonRank", rule)
        return _in_values("taxonRank", _BELOW_SPECIES_RANKS)
    if rule == "coordinates_present":
        _require(names, "decimalLatitude", rule)
        _require(names, "decimalLongitude", rule)
        return f"({_meaningful('decimalLatitude')}) AND ({_meaningful('decimalLongitude')})"
    if rule.startswith("event_date_has_"):
        _require(names, "eventDate", rule)
        patterns = {
            "event_date_has_year_precision": r"^[0-9]{4}",
            "event_date_has_month_precision": r"^[0-9]{4}-[0-9]{2}",
            "event_date_has_day_precision": r"^[0-9]{4}-[0-9]{2}-[0-9]{2}",
        }
        return (
            f"regexp_matches(trim(cast({_identifier('eventDate')} AS VARCHAR)), "
            f"{_literal(patterns[rule])})"
        )
    if rule == "direct_media_resource":
        _require(names, "media_identifier", rule)
        return (
            f"regexp_matches(trim(cast({_identifier('media_identifier')} AS VARCHAR)), "
            f"{_literal(r'(?i)^https?://')})"
        )
    raise ValueError(f"unknown applicability rule: {rule}")


def _nonblank(field: str) -> str:
    name = _identifier(field)
    return f"{name} IS NOT NULL AND trim(cast({name} AS VARCHAR)) <> ''"


def _meaningful(field: str) -> str:
    name = _identifier(field)
    sentinels = ", ".join(_literal(value) for value in _SEMANTIC_SENTINELS)
    return (
        f"({_nonblank(field)}) AND lower(trim(cast({name} AS VARCHAR))) "
        f"NOT IN ({sentinels})"
    )


def _in_values(field: str, values: tuple[str, ...]) -> str:
    options = ", ".join(_literal(value) for value in values)
    return (
        f"lower(trim(cast({_identifier(field)} AS VARCHAR))) IN ({options})"
    )


def _require(names: set[str], field: str, rule: str) -> None:
    if field not in names:
        raise ValueError(f"applicability rule {rule} requires field {field}")


def _percentage(numerator: int, denominator: int) -> float | None:
    return numerator * 100.0 / denominator if denominator else None


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


__all__ = [
    "COMPLETENESS_SCHEMA",
    "COMPLETENESS_SCHEMA_VERSION",
    "CompletenessProfile",
    "profile_completeness",
]
