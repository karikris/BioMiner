from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from biominer.storage.parquet import write_parquet


GBIF_API = "https://api.gbif.org/v1"
OPEN_TREE_API = "https://api.opentreeoflife.org/v3"
HIERARCHY_EVIDENCE_FILE = "hierarchy_evidence.parquet"
HIERARCHY_EVIDENCE_SCHEMA = {
    "accepted_taxon_id": pl.String,
    "accepted_scientific_name": pl.String,
    "target_rank": pl.String,
    "supplied_name": pl.String,
    "supplied_source_taxon_id": pl.String,
    "source": pl.String,
    "source_release": pl.String,
    "evidence_id": pl.String,
    "expected_family": pl.String,
    "matched_family": pl.String,
    "match_type": pl.String,
    "matched_status": pl.String,
    "confidence": pl.Float64,
    "previous_proxy_node_id": pl.String,
    "supersedes_node_id": pl.String,
    "applied": pl.Boolean,
    "rejection_reason": pl.String,
    "retrieved_at": pl.String,
}


async def harvest_gbif_genus_evidence(
    registry_dir: str | Path,
    output_dir: str | Path,
    *,
    workers: int = 8,
    max_retries: int = 4,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Resolve only missing genus proxies against GBIF with strict guardrails."""

    registry = Path(registry_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = pl.read_parquet(registry / "species_paths.parquet")
    taxa = pl.read_parquet(registry / "taxa.parquet")
    pending = paths.filter(pl.col("genus_candidate_kind") == "carry_forward_proxy")
    col_genera: dict[str, set[str]] = {}
    for row in taxa.filter(pl.col("rank") == "GENUS").select(["scientific_name", "family"]).iter_rows(named=True):
        col_genera.setdefault(str(row["scientific_name"]), set()).add(str(row["family"]))
    semaphore = asyncio.Semaphore(max(1, workers))
    limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)
    async with httpx.AsyncClient(
        base_url=GBIF_API,
        timeout=httpx.Timeout(timeout_seconds),
        limits=limits,
        headers={"User-Agent": "BioMiner/0.1 hierarchy-enrichment"},
    ) as client:
        results = await asyncio.gather(
            *(
                _match_proxy(
                    client,
                    semaphore,
                    dict(row),
                    col_genera=col_genera,
                    max_retries=max_retries,
                )
                for row in pending.iter_rows(named=True)
            )
        )
    frame = pl.DataFrame(results, schema=HIERARCHY_EVIDENCE_SCHEMA).sort(
        ["accepted_scientific_name", "accepted_taxon_id"]
    )
    write_parquet(frame, output / HIERARCHY_EVIDENCE_FILE)
    return {
        "evidence_path": str(output / HIERARCHY_EVIDENCE_FILE),
        "proxy_rows": pending.height,
        "applied_rows": frame.filter(pl.col("applied")).height,
        "rejected_rows": frame.filter(~pl.col("applied")).height,
        "rejections": _count_map(frame.filter(~pl.col("applied")), "rejection_reason"),
    }


async def harvest_open_tree_genus_evidence(
    registry_dir: str | Path,
    output_dir: str | Path,
    *,
    workers: int = 4,
    max_retries: int = 4,
    timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    """Try unresolved proxies against exact Open Tree TNRS lineages."""

    registry = Path(registry_dir)
    output = Path(output_dir)
    evidence_path = output / HIERARCHY_EVIDENCE_FILE
    existing = (
        pl.read_parquet(evidence_path).cast(HIERARCHY_EVIDENCE_SCHEMA)
        if evidence_path.is_file()
        else pl.DataFrame(schema=HIERARCHY_EVIDENCE_SCHEMA)
    )
    applied_ids = set(existing.filter(pl.col("applied"))["accepted_taxon_id"].to_list())
    paths = pl.read_parquet(registry / "species_paths.parquet")
    taxa = pl.read_parquet(registry / "taxa.parquet")
    pending = paths.filter(
        (pl.col("genus_candidate_kind") == "carry_forward_proxy")
        & (~pl.col("accepted_taxon_id").is_in(applied_ids))
    )
    col_genera: dict[str, set[str]] = {}
    for row in taxa.filter(pl.col("rank") == "GENUS").select(["scientific_name", "family"]).iter_rows(named=True):
        col_genera.setdefault(str(row["scientific_name"]), set()).add(str(row["family"]))
    if pending.is_empty():
        return {"evidence_path": str(evidence_path), "proxy_rows": 0, "applied_rows": 0, "rejected_rows": 0}
    names = pending["accepted_scientific_name"].to_list()
    limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)
    semaphore = asyncio.Semaphore(max(1, workers))
    async with httpx.AsyncClient(
        base_url=OPEN_TREE_API,
        timeout=httpx.Timeout(timeout_seconds),
        limits=limits,
        headers={"User-Agent": "BioMiner/0.1 hierarchy-enrichment"},
    ) as client:
        tnrs = await _post_json(
            client,
            semaphore,
            "/tnrs/match_names",
            payload={"names": names, "do_approximate_matching": False},
            max_retries=max_retries,
        )
        matches_by_name = {
            str(row.get("name") or ""): row.get("matches") or []
            for row in (tnrs.get("results") or [])
            if isinstance(row, dict)
        }
        results = await asyncio.gather(
            *(
                _open_tree_match_proxy(
                    client,
                    semaphore,
                    dict(row),
                    matches=matches_by_name.get(str(row["accepted_scientific_name"]), []),
                    col_genera=col_genera,
                    max_retries=max_retries,
                )
                for row in pending.iter_rows(named=True)
            )
        )
    additions = pl.DataFrame(results, schema=HIERARCHY_EVIDENCE_SCHEMA)
    combined = pl.concat([existing, additions], how="vertical_relaxed").unique(
        ["accepted_taxon_id", "source"], keep="last", maintain_order=True
    ).sort(["accepted_scientific_name", "source"])
    write_parquet(combined, evidence_path)
    return {
        "evidence_path": str(evidence_path),
        "proxy_rows": pending.height,
        "applied_rows": additions.filter(pl.col("applied")).height,
        "rejected_rows": additions.filter(~pl.col("applied")).height,
        "rejections": _count_map(additions.filter(~pl.col("applied")), "rejection_reason"),
    }


async def _match_proxy(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    path_row: dict[str, Any],
    *,
    col_genera: dict[str, set[str]],
    max_retries: int,
) -> dict[str, Any]:
    scientific_name = str(path_row["accepted_scientific_name"])
    expected_family = str(path_row["family"])
    payload = await _get_json(
        client,
        semaphore,
        "/species/match",
        params={"name": scientific_name, "rank": "SPECIES", "strict": "true"},
        max_retries=max_retries,
    )
    genus = str(payload.get("genus") or "")
    genus_key = str(payload.get("genusKey") or "")
    matched_family = str(payload.get("family") or "")
    match_type = str(payload.get("matchType") or "")
    matched_status = str(payload.get("status") or "")
    confidence = float(payload.get("confidence") or 0.0)
    rejection = ""
    if not genus or not genus_key:
        rejection = "gbif_missing_genus"
    elif matched_family != expected_family:
        rejection = "gbif_family_conflict"
    elif matched_status != "ACCEPTED":
        rejection = "gbif_status_not_accepted"
    elif confidence < 90 or match_type not in {"EXACT", "HIGHERRANK"}:
        rejection = "gbif_match_not_confident"
    elif genus in col_genera and expected_family not in col_genera[genus]:
        rejection = "col_genus_family_conflict"
    applied = not rejection
    previous_proxy = str(path_row.get("genus_node_id") or "")
    evidence_id = f"https://api.gbif.org/v1/species/match?name={scientific_name}"
    return {
        "accepted_taxon_id": str(path_row["accepted_taxon_id"]),
        "accepted_scientific_name": scientific_name,
        "target_rank": "GENUS",
        "supplied_name": genus,
        "supplied_source_taxon_id": f"gbif:{genus_key}" if genus_key else "",
        "source": "GBIF",
        "source_release": "GBIF Species API",
        "evidence_id": evidence_id,
        "expected_family": expected_family,
        "matched_family": matched_family,
        "match_type": match_type,
        "matched_status": matched_status,
        "confidence": confidence,
        "previous_proxy_node_id": previous_proxy,
        "supersedes_node_id": previous_proxy if applied else "",
        "applied": applied,
        "rejection_reason": rejection,
        "retrieved_at": datetime.now(UTC).isoformat(),
    }


async def _get_json(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    path: str,
    *,
    params: dict[str, str],
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                response = await client.get(path, params=params)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("GBIF match response must be an object")
            return payload
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            await asyncio.sleep(min(20.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"GBIF hierarchy request failed: {last_error}")


async def _open_tree_match_proxy(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    path_row: dict[str, Any],
    *,
    matches: list[dict[str, Any]],
    col_genera: dict[str, set[str]],
    max_retries: int,
) -> dict[str, Any]:
    exact = [
        match
        for match in matches
        if isinstance(match, dict)
        and float(match.get("score") or 0.0) == 1.0
        and str(match.get("matched_name") or "") == str(path_row["accepted_scientific_name"])
    ]
    genus = ""
    genus_id = ""
    family = ""
    source_release = "Open Tree Taxonomy"
    evidence_id = ""
    rejection = ""
    if len(exact) != 1:
        rejection = "open_tree_no_unique_exact_match"
    else:
        taxon = exact[0].get("taxon") or {}
        ott_id = str(taxon.get("ott_id") or "")
        source_release = str(taxon.get("source") or source_release)
        evidence_id = f"https://api.opentreeoflife.org/v3/taxonomy/taxon_info?ott_id={ott_id}"
        lineage_payload = await _post_json(
            client,
            semaphore,
            "/taxonomy/taxon_info",
            payload={"ott_id": int(ott_id), "include_lineage": True},
            max_retries=max_retries,
        )
        for ancestor in lineage_payload.get("lineage") or []:
            if not isinstance(ancestor, dict):
                continue
            if str(ancestor.get("rank") or "") == "genus" and not genus:
                genus = str(ancestor.get("name") or "")
                genus_id = f"ott:{ancestor.get('ott_id')}"
            if str(ancestor.get("rank") or "") == "family" and not family:
                family = str(ancestor.get("name") or "")
        if not genus or not genus_id:
            rejection = "open_tree_missing_genus"
        elif family != str(path_row["family"]):
            rejection = "open_tree_family_conflict"
        elif genus in col_genera and str(path_row["family"]) not in col_genera[genus]:
            rejection = "col_genus_family_conflict"
    applied = not rejection
    previous_proxy = str(path_row.get("genus_node_id") or "")
    return {
        "accepted_taxon_id": str(path_row["accepted_taxon_id"]),
        "accepted_scientific_name": str(path_row["accepted_scientific_name"]),
        "target_rank": "GENUS",
        "supplied_name": genus,
        "supplied_source_taxon_id": genus_id,
        "source": "Open Tree",
        "source_release": source_release,
        "evidence_id": evidence_id,
        "expected_family": str(path_row["family"]),
        "matched_family": family,
        "match_type": "EXACT" if exact else "NONE",
        "matched_status": "ACCEPTED" if exact else "",
        "confidence": 100.0 if exact else 0.0,
        "previous_proxy_node_id": previous_proxy,
        "supersedes_node_id": previous_proxy if applied else "",
        "applied": applied,
        "rejection_reason": rejection,
        "retrieved_at": datetime.now(UTC).isoformat(),
    }


async def _post_json(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    path: str,
    *,
    payload: dict[str, Any],
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                response = await client.post(path, json=payload)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Open Tree response must be an object")
            return result
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            await asyncio.sleep(min(20.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"Open Tree hierarchy request failed: {last_error}")


def _count_map(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if frame.is_empty():
        return {}
    return {
        str(row[column]): int(row["len"])
        for row in frame.group_by(column).len().sort(column).iter_rows(named=True)
    }


__all__ = [
    "HIERARCHY_EVIDENCE_FILE",
    "HIERARCHY_EVIDENCE_SCHEMA",
    "harvest_gbif_genus_evidence",
    "harvest_open_tree_genus_evidence",
]
