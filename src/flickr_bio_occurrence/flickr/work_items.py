from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date
from hashlib import sha256
from typing import Iterable


QUERY_VARIANTS = (
    "scientific_name",
    "lime_butterfly",
    "chequered_swallowtail",
    "citrus_swallowtail",
    "swallowtail",
)
BROAD_QUERY_VARIANTS = QUERY_VARIANTS + (
    "papilio",
    "butterfly",
    "citrusbutterfly",
    "limebutterfly",
)


@dataclass(frozen=True)
class WorkItem:
    species_name: str
    species_query_terms: list[str]
    region_id: str
    region_name: str
    bbox: str
    year: int
    month: int
    min_taken_date: str
    max_taken_date: str
    page: int
    query_variant: str
    status: str = "pending"
    attempt_count: int = 0
    last_error: str | None = None
    work_item_id: str = field(init=False)

    def __post_init__(self) -> None:
        identity = f"{self.species_name}{self.region_id}{self.year}{self.month}{self.page}{self.query_variant}"
        object.__setattr__(self, "work_item_id", sha256(identity.encode("utf-8")).hexdigest())


def build_monthly_work_items(
    *,
    species_name: str,
    species_query_terms: list[str],
    regions: Iterable[tuple[str, str, str]],
    years: Iterable[int],
    months: Iterable[int],
    query_variants: Iterable[str] | None = None,
    pages: Iterable[int] | None = None,
    page: int = 1,
    end_date: date | None = None,
) -> list[WorkItem]:
    items: list[WorkItem] = []
    variants = tuple(query_variants or QUERY_VARIANTS)
    page_values = tuple(pages or [page])
    for region_id, region_name, bbox in regions:
        for year in years:
            for month in months:
                if end_date and (year, month) > (end_date.year, end_date.month):
                    continue
                last_day = monthrange(year, month)[1]
                max_taken_date = f"{year}-{month:02d}-{last_day:02d}"
                if end_date and (year, month) == (end_date.year, end_date.month):
                    max_taken_date = end_date.isoformat()
                for query_variant in variants:
                    for page_value in page_values:
                        items.append(
                            WorkItem(
                                species_name=species_name,
                                species_query_terms=species_query_terms,
                                region_id=region_id,
                                region_name=region_name,
                                bbox=bbox,
                                year=year,
                                month=month,
                                min_taken_date=f"{year}-{month:02d}-01",
                                max_taken_date=max_taken_date,
                                page=page_value,
                                query_variant=query_variant,
                            )
                        )
    return items
