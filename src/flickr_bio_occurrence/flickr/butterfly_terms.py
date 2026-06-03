from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import polars as pl


DEFAULT_COMMON_BUTTERFLY_TERMS = (
    "butterfly",
    "butterflies",
    "skipper",
    "grass-dart",
    "dart",
    "swallowtail",
    "jezebel",
    "blue butterfly",
    "copper butterfly",
    "brown butterfly",
    "azure butterfly",
    "hairstreak",
)


@dataclass(frozen=True)
class ButterflySearchTerm:
    term: str
    source: str


def load_butterfly_dashboard_terms(
    dashboard_data_dir: str | Path = "/home/toffe/butterfly-dashboard/data",
    *,
    common_terms: tuple[str, ...] = DEFAULT_COMMON_BUTTERFLY_TERMS,
) -> list[ButterflySearchTerm]:
    data_dir = Path(dashboard_data_dir)
    parquet_path = data_dir / "butterfly_sa2_bins.parquet"
    frame = pl.scan_parquet(parquet_path).select(["scientificName", "species", "genus", "family"]).unique().collect()
    terms: dict[str, str] = {}
    for source in ("scientificName", "species", "genus", "family"):
        for value in frame[source].drop_nulls().unique().to_list():
            _add_term(terms, str(value), source)
    for value in _common_names_from_reference(data_dir / "reference" / "butterfly_conservation_status.csv"):
        _add_term(terms, value, "common_name")
    for value in common_terms:
        _add_term(terms, value, "common_word")
    return [
        ButterflySearchTerm(term=term, source=source)
        for term, source in sorted(terms.items(), key=lambda item: (_source_rank(item[1]), item[0].casefold()))
    ]


def estimate_minimum_fetch_hours(*, planned_api_calls: int, api_calls_per_hour: int = 3600) -> float:
    if planned_api_calls <= 0:
        return 0.0
    return planned_api_calls / api_calls_per_hour


def safe_query_variant(term: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", term.casefold()).strip("_")
    return slug or "query"


def _add_term(terms: dict[str, str], value: str, source: str) -> None:
    term = " ".join(value.split())
    if not term:
        return
    terms.setdefault(term, source)


def _common_names_from_reference(path: Path) -> list[str]:
    if not path.exists():
        return []
    names: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("common_name") or ""
            for part in raw.replace("/", "|").split("|"):
                name = " ".join(part.split())
                if name:
                    names.append(name)
    return names


def _source_rank(source: str) -> int:
    return {
        "scientificName": 0,
        "species": 1,
        "common_name": 2,
        "genus": 3,
        "family": 4,
        "common_word": 5,
    }.get(source, 99)
