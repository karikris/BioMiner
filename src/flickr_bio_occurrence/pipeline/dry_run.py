from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any

from flickr_bio_occurrence.flickr.work_items import build_monthly_work_items


REGIONS = {
    "AU_ALL": ("AU_ALL", "Australia", "112.92,-43.74,153.64,-10.05"),
    "AU_QLD": ("AU_QLD", "Queensland", "137.99,-29.18,153.55,-9.14"),
}
PAPILIO_DEMOLEUS_TERMS = [
    "Papilio demoleus",
    "lime butterfly",
    "chequered swallowtail",
    "citrus swallowtail",
    "swallowtail",
]


def build_dry_run_summary(
    *,
    species: str,
    region: str,
    year: int,
    month: int,
    config_path: str | Path,
    pages: range | None = None,
) -> dict[str, Any]:
    config = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    region_tuple = REGIONS[region]
    work_items = build_monthly_work_items(
        species_name=species,
        species_query_terms=PAPILIO_DEMOLEUS_TERMS,
        regions=[region_tuple],
        years=[year],
        months=[month],
        pages=pages,
    )
    per_page = int(config["flickr"]["default_per_page"])
    planned_max_records = min(
        len(work_items) * per_page,
        int(config["flickr"]["hard_photo_records_per_hour"]),
    )
    planned_api_calls = len(work_items)
    soft_cap = int(config["flickr"]["soft_api_calls_per_hour"])
    return {
        "species": species,
        "region": region,
        "year": year,
        "month": month,
        "planned_api_calls": planned_api_calls,
        "planned_maximum_photo_records": planned_max_records,
        "hourly_limit_status": "within_soft_cap" if planned_api_calls < soft_cap else "at_or_over_soft_cap",
        "work_item_count": len(work_items),
        "output_paths": {
            "raw": "data/raw/flickr/photos_search/",
            "triage": "data/image_triage/",
            "review": "data/review/",
        },
        "vision_package": "BioCLIP 2.5 register runner",
    }
