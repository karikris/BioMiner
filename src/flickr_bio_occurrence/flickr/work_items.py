from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Iterable

from flickr_bio_occurrence.taxonomy.species_mapper import SpeciesSeed


QUERY_VARIANTS = ("scientific_name", "lime_butterfly", "chequered_swallowtail", "citrus_swallowtail", "swallowtail")


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
    species: SpeciesSeed,
    regions: Iterable[tuple[str, str, str]],
    years: Iterable[int],
    months: Iterable[int],
    query_variants: Iterable[str] | None = None,
    page: int = 1,
) -> list[WorkItem]:
    items: list[WorkItem] = []
    variants = tuple(query_variants or QUERY_VARIANTS)
    for region_id, region_name, bbox in regions:
        for year in years:
            for month in months:
                last_day = monthrange(year, month)[1]
                for query_variant in variants:
                    items.append(
                        WorkItem(
                            species_name=species.accepted_scientific_name,
                            species_query_terms=species.search_terms,
                            region_id=region_id,
                            region_name=region_name,
                            bbox=bbox,
                            year=year,
                            month=month,
                            min_taken_date=f"{year}-{month:02d}-01",
                            max_taken_date=f"{year}-{month:02d}-{last_day:02d}",
                            page=page,
                            query_variant=query_variant,
                        )
                    )
    return items
