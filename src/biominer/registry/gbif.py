from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import httpx


GBIF_BASE_URL = "https://api.gbif.org/v1"
JSONPayload = dict[str, Any] | list[dict[str, Any]]
HTTPGet = Callable[[str, dict[str, object]], JSONPayload]


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
        return self._get_object("/species/match", params)

    def usage(self, key: int | str) -> dict[str, Any]:
        return self._get_object(f"/species/{key}", {})

    def parents(self, key: int | str) -> list[dict[str, Any]]:
        payload = self._get(f"/species/{key}/parents", {})
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        return _results(payload)

    def children(self, key: int | str, *, rank: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        params: dict[str, object] = {"limit": limit}
        if rank:
            params["rank"] = rank
        return self._paginated_results(f"/species/{key}/children", params, limit=limit)

    def synonyms(self, key: int | str, *, limit: int = 1000) -> list[dict[str, Any]]:
        return self._paginated_results(f"/species/{key}/synonyms", {"limit": limit}, limit=limit)

    def vernacular_names(self, key: int | str, *, limit: int = 1000) -> list[dict[str, Any]]:
        return self._paginated_results(f"/species/{key}/vernacularNames", {"limit": limit}, limit=limit)

    def occurrence_search(self, params: dict[str, object]) -> dict[str, Any]:
        return self._get_object("/occurrence/search", params)

    def _get(self, path: str, params: dict[str, object]) -> JSONPayload:
        self.call_count += 1
        return self._http_get(path, params)

    def _get_object(self, path: str, params: dict[str, object]) -> dict[str, Any]:
        payload = self._get(path, params)
        if not isinstance(payload, dict):
            raise ValueError(f"GBIF response for {path} must be a JSON object")
        return payload

    def _paginated_results(self, path: str, params: dict[str, object], *, limit: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page_params = dict(params)
            if offset:
                page_params["offset"] = offset
            payload = self._get_object(path, page_params)
            page_rows = _results(payload)
            rows.extend(page_rows)
            if _is_final_page(payload, rows_returned=len(page_rows), offset=offset, limit=limit):
                return rows
            offset += len(page_rows)


def resolve_family(
    client: GBIFClient,
    family_name: str,
    *,
    root_name: str,
    expected_taxon_key: int | str | None = None,
) -> FamilyResolution:
    match = client.match_name(family_name, rank="FAMILY")

    matched_usage_key = match.get("usageKey")
    match_candidate_key = match.get("acceptedUsageKey") or matched_usage_key
    match_type = str(match.get("matchType") or "")
    confidence = int(match.get("confidence") or 0)
    match_rank = str(match.get("rank") or "")

    if expected_taxon_key is not None:
        accepted_key = str(expected_taxon_key).removeprefix("gbif:")

        # A valid conflicting family match is a production error.
        if (
            match_candidate_key is not None
            and match_rank == "FAMILY"
            and match_type not in {"", "NONE", "HIGHERRANK"}
            and str(match_candidate_key) != accepted_key
        ):
            raise ValueError(
                f"GBIF family {family_name!r} matched family key "
                f"{match_candidate_key!r}, but configured key is "
                f"{accepted_key!r}"
            )
    else:
        if match_candidate_key is None:
            raise ValueError(f"GBIF did not resolve family {family_name!r}")
        accepted_key = str(match_candidate_key)

    usage = client.usage(accepted_key)

    rank = str(usage.get("rank") or "")
    if rank != "FAMILY":
        raise ValueError(
            f"GBIF family {family_name!r} key {accepted_key!r} "
            f"resolved to rank {rank!r}, expected rank FAMILY"
        )

    resolved_name = str(
        usage.get("canonicalName")
        or usage.get("scientificName")
        or ""
    )
    # For configured production families, the pinned key must resolve to
    # the exact reviewed family name. For unpinned synonym lookups, the
    # accepted usage name is expected to differ from the submitted synonym.
    if expected_taxon_key is not None and resolved_name != family_name:
        raise ValueError(
            f"GBIF family key {accepted_key!r} resolved to "
            f"{resolved_name!r}, expected {family_name!r}"
        )

    status = str(
        usage.get("taxonomicStatus")
        or usage.get("status")
        or ""
    )
    if status and status != "ACCEPTED":
        raise ValueError(
            f"GBIF family {family_name!r} key {accepted_key!r} "
            f"has status {status!r}, expected ACCEPTED"
        )

    lineage = tuple(_lineage_names(usage))
    if not lineage:
        lineage = tuple(
            str(
                parent.get("canonicalName")
                or parent.get("scientificName")
                or ""
            )
            for parent in client.parents(accepted_key)
            if parent.get("canonicalName")
            or parent.get("scientificName")
        )

    # GBIF currently omits the Papilionoidea superfamily node from the
    # backbone parent chain of these families. Require Lepidoptera when
    # the configured superfamily is absent.
    if root_name not in lineage and "Lepidoptera" not in lineage:
        raise ValueError(
            f"GBIF family {family_name!r} lineage contains neither "
            f"{root_name!r} nor order 'Lepidoptera': {lineage!r}"
        )

    return FamilyResolution(
        accepted_taxon_key=f"gbif:{accepted_key}",
        matched_usage_key=(
            f"gbif:{matched_usage_key}"
            if matched_usage_key is not None
            else ""
        ),
        scientific_name=resolved_name,
        match_type=match_type,
        confidence=confidence,
        lineage_names=lineage,
    )


def _http_get(path: str, params: dict[str, object]) -> JSONPayload:
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{GBIF_BASE_URL}{path}", params=params)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        return payload
    raise ValueError(
        f"GBIF response for {path} must be a JSON object or an array of objects"
    )


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
