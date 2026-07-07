from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.registry.normalize import normalize_name_key


QUERY_CURATION_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class QueryCurationRule:
    accepted_taxon_key: str
    normalized_match_key: str
    source: str
    action: str
    reason: str


def load_query_curation_rules(path: str | Path | None) -> tuple[QueryCurationRule, ...]:
    if path is None:
        return ()
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"query curation JSON does not exist: {candidate}")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if str(payload.get("schema_version") or "") != QUERY_CURATION_SCHEMA_VERSION:
        raise ValueError(f"query curation schema_version must be {QUERY_CURATION_SCHEMA_VERSION}")
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list):
        raise ValueError("query curation rules must be a JSON list")
    rules: list[QueryCurationRule] = []
    for index, item in enumerate(raw_rules):
        if not isinstance(item, dict):
            raise ValueError(f"query curation rule {index} must be an object")
        action = str(item.get("action") or "")
        if action != "disable_query":
            raise ValueError(f"unsupported query curation action: {action}")
        accepted_taxon_key = str(item.get("accepted_taxon_key") or "")
        normalized_match_key = normalize_name_key(item.get("normalized_match_key") or item.get("display_name") or "")
        reason = str(item.get("reason") or "")
        if not accepted_taxon_key or not normalized_match_key or not reason:
            raise ValueError(f"query curation rule {index} requires accepted_taxon_key, normalized_match_key, and reason")
        rules.append(
            QueryCurationRule(
                accepted_taxon_key=accepted_taxon_key,
                normalized_match_key=normalized_match_key,
                source=str(item.get("source") or ""),
                action=action,
                reason=reason,
            )
        )
    return tuple(rules)


def apply_query_curation(names: pl.DataFrame, rules: tuple[QueryCurationRule, ...]) -> pl.DataFrame:
    if names.is_empty() or not rules:
        return names
    rows: list[dict[str, Any]] = []
    for row in names.to_dicts():
        rule = _matching_rule(row, rules)
        if rule is None:
            rows.append(row)
            continue
        current_score = float(row.get("species_specificity_score") or 0.0)
        rows.append(
            {
                **row,
                "query_eligible": False,
                "query_disabled_reason": rule.reason,
                "species_specificity_score": min(current_score, 0.45),
            }
        )
    return pl.DataFrame(rows, schema=names.schema)


def _matching_rule(row: dict[str, Any], rules: tuple[QueryCurationRule, ...]) -> QueryCurationRule | None:
    accepted_taxon_key = str(row.get("accepted_taxon_key") or "")
    normalized_match_key = str(row.get("normalized_match_key") or normalize_name_key(row.get("display_name") or row.get("verbatim_name") or ""))
    source = _source_key(row.get("source"))
    for rule in rules:
        if rule.accepted_taxon_key != accepted_taxon_key:
            continue
        if rule.normalized_match_key != normalized_match_key:
            continue
        if rule.source and _source_key(rule.source) != source:
            continue
        return rule
    return None


def _source_key(value: object) -> str:
    return "_".join(str(value or "").casefold().split())


__all__ = ["QueryCurationRule", "apply_query_curation", "load_query_curation_rules"]
