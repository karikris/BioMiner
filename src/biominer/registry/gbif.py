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

    def match_name(self, name: str, *, rank: str | None = None, strict: bool = False) -> dict[str, Any]:
        params: dict[str, object] = {"name": name, "strict": str(strict).lower()}
        if rank:
            params["rank"] = rank
        return self._http_get("/species/match", params)

    def usage(self, key: int | str) -> dict[str, Any]:
        return self._http_get(f"/species/{key}", {})

    def children(self, key: int | str, *, rank: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        params: dict[str, object] = {"limit": limit}
        if rank:
            params["rank"] = rank
        return _results(self._http_get(f"/species/{key}/children", params))

    def synonyms(self, key: int | str, *, limit: int = 1000) -> list[dict[str, Any]]:
        return _results(self._http_get(f"/species/{key}/synonyms", {"limit": limit}))

    def vernacular_names(self, key: int | str, *, limit: int = 1000) -> list[dict[str, Any]]:
        return _results(self._http_get(f"/species/{key}/vernacularNames", {"limit": limit}))


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


def _lineage_names(usage: dict[str, Any]) -> list[str]:
    parents = usage.get("parents", [])
    if isinstance(parents, list):
        return [str(parent.get("scientificName")) for parent in parents if isinstance(parent, dict) and parent.get("scientificName")]
    higher = usage.get("higherClassificationMap", {})
    if isinstance(higher, dict):
        return [str(value) for value in higher.values()]
    return []
