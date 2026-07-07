from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.registry.normalize import parse_language_tag


COUNTRY_LANGUAGE_SCHEMA_VERSION = "country-language-overrides-v1"
SPECIES_REGION_LANGUAGE_SCHEMA_VERSION = "species-region-language-targets-v1"
COUNTRY_LANGUAGE_TARGETS_FILE = "country_language_targets.parquet"


@dataclass(frozen=True)
class LanguageTargetSpec:
    language: str
    language_name: str
    priority: int
    source: str
    source_version: str
    reason: str


def generate_language_targets(
    range_countries: pl.DataFrame,
    *,
    country_language_overrides_json: str | Path | None = None,
    species_region_language_targets_json: str | Path | None = None,
) -> pl.DataFrame:
    if range_countries.is_empty():
        return _empty_targets()
    country_specs = _load_country_language_specs(country_language_overrides_json)
    region_specs = _load_region_language_specs(species_region_language_targets_json)
    rows: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for range_row in range_countries.to_dicts():
        country_code = str(range_row.get("country_code") or "").upper()
        region = str(range_row.get("region") or "")
        specs = [*country_specs.get(country_code, ()), *region_specs.get(region, ())]
        for spec in specs:
            target = _target_row(range_row, spec)
            key = (
                str(target["accepted_taxon_key"]),
                str(target["country_code"]),
                str(target["admin1_code"]),
                str(target["language_code"]),
            )
            existing = rows.get(key)
            if existing is None or int(target["priority"]) < int(existing["priority"]):
                rows[key] = target
    if not rows:
        return _empty_targets()
    return pl.DataFrame(list(rows.values()), schema=_target_schema()).sort(
        ["accepted_taxon_key", "country_code", "admin1_code", "priority", "language_code"]
    )


def write_language_targets(frame: pl.DataFrame, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / COUNTRY_LANGUAGE_TARGETS_FILE
    frame.write_parquet(path)
    return path


def _target_row(range_row: dict[str, Any], spec: LanguageTargetSpec) -> dict[str, object]:
    tag = parse_language_tag(spec.language)
    enabled = bool(range_row.get("range_status") != "taxonomically_cautionary" and not bool(range_row.get("taxonomic_caution")))
    return {
        "accepted_taxon_key": str(range_row.get("accepted_taxon_key") or ""),
        "scientific_name": str(range_row.get("scientific_name") or ""),
        "country_code": str(range_row.get("country_code") or ""),
        "country_name": str(range_row.get("country_name") or ""),
        "admin1_code": str(range_row.get("admin1_code") or ""),
        "admin1_name": str(range_row.get("admin1_name") or ""),
        "language_code": tag.language,
        "language_name": spec.language_name,
        "script": tag.script,
        "region": str(range_row.get("region") or ""),
        "priority": spec.priority,
        "reason": spec.reason,
        "source": spec.source,
        "source_version": spec.source_version,
        "enabled": enabled,
        "disabled_reason": "" if enabled else "taxonomic_caution_range",
    }


def _load_country_language_specs(path: str | Path | None) -> dict[str, tuple[LanguageTargetSpec, ...]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(payload.get("schema_version") or "") != COUNTRY_LANGUAGE_SCHEMA_VERSION:
        raise ValueError(f"country language config schema_version must be {COUNTRY_LANGUAGE_SCHEMA_VERSION}")
    source = str(payload.get("source") or "country_language_overrides")
    source_version = str(payload.get("source_version") or "")
    result: dict[str, list[LanguageTargetSpec]] = {}
    for country in payload.get("countries") or []:
        if not isinstance(country, dict):
            continue
        country_code = str(country.get("country_code") or "").upper()
        specs = _language_specs(
            country.get("languages") or (),
            source=source,
            source_version=source_version,
            reason="country_language_override",
            default_priority=50,
        )
        admin1_specs = _admin1_language_specs(country, source=source, source_version=source_version)
        result.setdefault(country_code, []).extend([*specs, *admin1_specs])
    return {code: tuple(specs) for code, specs in result.items() if code}


def _admin1_language_specs(country: dict[str, Any], *, source: str, source_version: str) -> tuple[LanguageTargetSpec, ...]:
    specs: list[LanguageTargetSpec] = []
    for admin1 in country.get("admin1_languages") or []:
        if not isinstance(admin1, dict):
            continue
        specs.extend(
            _language_specs(
                admin1.get("languages") or (),
                source=source,
                source_version=source_version,
                reason="country_language_override",
                default_priority=45,
            )
        )
    return tuple(specs)


def _load_region_language_specs(path: str | Path | None) -> dict[str, tuple[LanguageTargetSpec, ...]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(payload.get("schema_version") or "") != SPECIES_REGION_LANGUAGE_SCHEMA_VERSION:
        raise ValueError(f"species region language config schema_version must be {SPECIES_REGION_LANGUAGE_SCHEMA_VERSION}")
    source = str(payload.get("source") or "species_region_language_targets")
    source_version = str(payload.get("source_version") or "")
    result: dict[str, tuple[LanguageTargetSpec, ...]] = {}
    for region in payload.get("regions") or []:
        if not isinstance(region, dict):
            continue
        region_name = str(region.get("region") or "")
        result[region_name] = _language_specs(
            region.get("languages") or (),
            source=source,
            source_version=source_version,
            reason="species_region_language_target",
            default_priority=60,
        )
    return result


def _language_specs(
    items: Any,
    *,
    source: str,
    source_version: str,
    reason: str,
    default_priority: int,
) -> tuple[LanguageTargetSpec, ...]:
    specs: list[LanguageTargetSpec] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        language = str(item.get("language") or "")
        if not language:
            continue
        specs.append(
            LanguageTargetSpec(
                language=language,
                language_name=str(item.get("language_name") or language),
                priority=int(item.get("priority") or default_priority + index),
                source=str(item.get("source") or source),
                source_version=str(item.get("source_version") or source_version),
                reason=reason,
            )
        )
    return tuple(specs)


def _empty_targets() -> pl.DataFrame:
    return pl.DataFrame([], schema=_target_schema())


def _target_schema() -> dict[str, pl.DataType]:
    return {
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "country_code": pl.String,
        "country_name": pl.String,
        "admin1_code": pl.String,
        "admin1_name": pl.String,
        "language_code": pl.String,
        "language_name": pl.String,
        "script": pl.String,
        "region": pl.String,
        "priority": pl.Int64,
        "reason": pl.String,
        "source": pl.String,
        "source_version": pl.String,
        "enabled": pl.Boolean,
        "disabled_reason": pl.String,
    }


__all__ = [
    "COUNTRY_LANGUAGE_TARGETS_FILE",
    "LanguageTargetSpec",
    "generate_language_targets",
    "write_language_targets",
]
