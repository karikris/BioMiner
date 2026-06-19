from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import httpx


GBIF_BASE_URL = "https://api.gbif.org/v1"
HTTPGet = Callable[[str, dict[str, object]], dict[str, Any]]


@dataclass(frozen=True)
class FamilyResolution:
    accepted_taxon_key: str
    matched_usage_key: str
    scientific_name: str
    match_type: str
    confidence: int
    lineage_names: tuple[str, ...]


class GBIFClient:
    def __init__(self, *, http_get: HTTPGet | None = None, base_url: str = GBIF_BASE_URL) -> None:
        self._http_get = http_get or _http_get
        self.base_url = base_url.rstrip("/")
        self.call_count = 0

    def match_name(self, name: str, *, rank: str | None = None, strict: bool = False) -> dict[str, Any]:
        params: dict[str, object] = {"name": name, "strict": str(strict).lower()}
        if rank:
            params["rank"] = rank
        return self._get("/species/match", params)

    def usage(self, key: int | str) -> dict[str, Any]:
        return self._get(f"/species/{key}", {})

    def children(self, key: int | str, *, rank: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        params: dict[str, object] = {"limit": limit}
        if rank:
            params["rank"] = rank
        return self._paginated_results(f"/species/{key}/children", params, limit=limit)

    def synonyms(self, key: int | str, *, limit: int = 1000) -> list[dict[str, Any]]:
        return self._paginated_results(f"/species/{key}/synonyms", {"limit": limit}, limit=limit)

    def vernacular_names(self, key: int | str, *, limit: int = 1000) -> list[dict[str, Any]]:
        return self._paginated_results(f"/species/{key}/vernacularNames", {"limit": limit}, limit=limit)

    def _get(self, path: str, params: dict[str, object]) -> dict[str, Any]:
        self.call_count += 1
        return self._http_get(path, params)

    def _paginated_results(self, path: str, params: dict[str, object], *, limit: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page_params = dict(params)
            if offset:
                page_params["offset"] = offset
            payload = self._get(path, page_params)
            page_rows = _results(payload)
            rows.extend(page_rows)
            if _is_final_page(payload, rows_returned=len(page_rows), offset=offset, limit=limit):
                return rows
            offset += len(page_rows)


def resolve_family(client: GBIFClient, family_name: str, *, root_name: str) -> FamilyResolution:
    match = client.match_name(family_name, rank="FAMILY")
    usage_key = match.get("usageKey")
    if usage_key is None:
        raise ValueError(f"GBIF did not resolve family {family_name!r}")
    accepted_key = match.get("acceptedUsageKey") or usage_key
    usage = client.usage(accepted_key)
    rank = str(usage.get("rank") or match.get("rank") or "")
    if rank != "FAMILY":
        raise ValueError(f"GBIF family {family_name!r} resolved to rank {rank!r}, expected rank FAMILY")
    lineage = tuple(_lineage_names(usage))
    if root_name not in lineage:
        raise ValueError(f"GBIF family {family_name!r} lineage does not contain {root_name}")
    return FamilyResolution(
        accepted_taxon_key=f"gbif:{accepted_key}",
        matched_usage_key=f"gbif:{usage_key}",
        scientific_name=str(usage.get("scientificName") or match.get("scientificName") or family_name),
        match_type=str(match.get("matchType") or ""),
        confidence=int(match.get("confidence") or 0),
        lineage_names=lineage,
    )


def _http_get(path: str, params: dict[str, object]) -> dict[str, Any]:
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{GBIF_BASE_URL}{path}", params=params)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"GBIF response for {path} must be a JSON object")
    return payload


def _results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("results", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _is_final_page(payload: dict[str, Any], *, rows_returned: int, offset: int, limit: int) -> bool:
    if payload.get("endOfRecords") is True:
        return True
    if rows_returned == 0:
        return True
    count = payload.get("count")
    if isinstance(count, int) and offset + rows_returned >= count:
        return True
    return rows_returned < limit


def _lineage_names(usage: dict[str, Any]) -> list[str]:
    parents = usage.get("parents", [])
    if isinstance(parents, list):
        return [str(parent.get("scientificName")) for parent in parents if isinstance(parent, dict) and parent.get("scientificName")]
    higher = usage.get("higherClassificationMap", {})
    if isinstance(higher, dict):
        return [str(value) for value in higher.values()]
    return []
