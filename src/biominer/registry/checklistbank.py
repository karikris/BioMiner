from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from biominer.registry.scope import load_scope
from biominer.registry.unified import COL_XR_DATASET_KEY, COL_XR_DOI, COL_XR_RELEASE
from biominer.storage.parquet import write_parquet


CHECKLISTBANK_API = "https://api.checklistbank.org"
COL_XR_ROOT_ID = "5G9"
SOURCE_NODES_FILE = "col_xr_nodes.parquet"
SOURCE_NAMES_FILE = "col_xr_names.parquet"
SOURCE_SNAPSHOTS_FILE = "source_snapshots.parquet"
SOURCE_NAME_STATE_FILE = "col_xr_name_harvest_state.parquet"

_NODE_SCHEMA = {
    "source_taxon_id": pl.String,
    "parent_source_taxon_id": pl.String,
    "scientific_name": pl.String,
    "scientific_name_authorship": pl.String,
    "rank": pl.String,
    "source_status": pl.String,
    "child_count": pl.Int64,
    "expanded": pl.Boolean,
    "family_source_taxon_id": pl.String,
    "family": pl.String,
    "genus_source_taxon_id": pl.String,
    "genus": pl.String,
}
_NAME_SCHEMA = {
    "accepted_source_taxon_id": pl.String,
    "display_name": pl.String,
    "language": pl.String,
    "name_class": pl.String,
    "source": pl.String,
    "source_record_id": pl.String,
    "trust_tier": pl.String,
    "review_state": pl.String,
    "enabled": pl.Boolean,
}
_SNAPSHOT_SCHEMA = {
    "source": pl.String,
    "source_version": pl.String,
    "source_dataset_key": pl.String,
    "doi": pl.String,
    "source_url": pl.String,
    "retrieved_at": pl.String,
    "transport": pl.String,
}
_NAME_STATE_SCHEMA = {
    "source_taxon_id": pl.String,
    "status": pl.String,
    "attempts": pl.Int64,
    "last_error": pl.String,
    "updated_at": pl.String,
}


async def harvest_col_xr_taxonomy(
    output_dir: str | Path,
    *,
    scope_path: str | Path = "config/butterfly_scope.json",
    workers: int = 16,
    max_retries: int = 6,
    timeout_seconds: float = 90.0,
) -> dict[str, Any]:
    """Harvest the pinned Papilionoidea scope through the public CLB tree API.

    The harvester writes only Parquet checkpoints. Re-running it resumes from
    nodes whose children have not yet been expanded.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scope = load_scope(scope_path)
    nodes_path = output / SOURCE_NODES_FILE
    names_path = output / SOURCE_NAMES_FILE
    nodes = _read_typed(nodes_path, _NODE_SCHEMA)
    names = _read_typed(names_path, _NAME_SCHEMA)
    if nodes.is_empty():
        nodes = _seed_nodes(scope.included_families)
        names = _accepted_name_rows(nodes)
        _write_source(nodes, names, output)

    limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)
    headers = {"User-Agent": "BioMiner/0.1 registry-harvester"}
    semaphore = asyncio.Semaphore(max(1, workers))
    async with httpx.AsyncClient(
        base_url=CHECKLISTBANK_API,
        timeout=httpx.Timeout(timeout_seconds),
        limits=limits,
        headers=headers,
    ) as client:
        while True:
            pending = nodes.filter(
                (~pl.col("expanded"))
                & (pl.col("child_count") > 0)
                & (pl.col("rank") != "SPECIES")
            )
            if pending.is_empty():
                break
            batch = pending.head(max(1, workers * 2)).to_dicts()
            results = await asyncio.gather(
                *(
                    _fetch_children(
                        client,
                        semaphore,
                        str(row["source_taxon_id"]),
                        max_retries=max_retries,
                    )
                    for row in batch
                )
            )
            expanded_ids = {str(row["source_taxon_id"]) for row in batch}
            nodes = nodes.with_columns(
                pl.when(pl.col("source_taxon_id").is_in(expanded_ids))
                .then(pl.lit(True))
                .otherwise(pl.col("expanded"))
                .alias("expanded")
            )
            child_rows = [child for children in results for child in children]
            if child_rows:
                additions = _node_frame(child_rows)
                nodes = pl.concat([nodes, additions], how="vertical_relaxed").unique(
                    "source_taxon_id", keep="first", maintain_order=True
                )
            nodes = _attach_lineage(nodes)
            names = pl.concat([names, _accepted_name_rows(nodes)], how="vertical_relaxed").unique(
                ["accepted_source_taxon_id", "display_name", "language", "name_class", "source"],
                keep="first",
                maintain_order=True,
            )
            _write_source(nodes, names, output)

    snapshot = pl.DataFrame(
        [
            {
                "source": "CoL XR",
                "source_version": COL_XR_RELEASE,
                "source_dataset_key": COL_XR_DATASET_KEY,
                "doi": COL_XR_DOI,
                "source_url": f"{CHECKLISTBANK_API}/dataset/{COL_XR_DATASET_KEY}",
                "retrieved_at": datetime.now(UTC).isoformat(),
                "transport": "ChecklistBank public tree API",
            }
        ],
        schema=_SNAPSHOT_SCHEMA,
    )
    write_parquet(snapshot, output / SOURCE_SNAPSHOTS_FILE)
    return {
        "source_dir": str(output),
        "taxon_rows": nodes.height,
        "species_rows": nodes.filter(pl.col("rank") == "SPECIES").height,
        "genus_rows": nodes.filter(pl.col("rank") == "GENUS").height,
        "family_rows": nodes.filter(pl.col("rank") == "FAMILY").height,
        "name_rows": names.height,
        "pending_expansions": nodes.filter(
            (~pl.col("expanded")) & (pl.col("child_count") > 0) & (pl.col("rank") != "SPECIES")
        ).height,
    }


async def harvest_col_xr_names(
    source_dir: str | Path,
    *,
    workers: int = 32,
    max_retries: int = 4,
    timeout_seconds: float = 45.0,
    batch_size: int = 256,
    limit: int = 0,
) -> dict[str, Any]:
    """Append CoL synonyms and vernaculars to the Parquet source snapshot."""

    source = Path(source_dir)
    nodes = _read_typed(source / SOURCE_NODES_FILE, _NODE_SCHEMA)
    names = _read_typed(source / SOURCE_NAMES_FILE, _NAME_SCHEMA)
    state = _read_typed(source / SOURCE_NAME_STATE_FILE, _NAME_STATE_SCHEMA)
    completed = set(
        state.filter(pl.col("status") == "complete")["source_taxon_id"].to_list()
    )
    target = nodes.filter(pl.col("rank").is_in(["FAMILY", "GENUS", "SPECIES"]))
    pending = [
        str(value)
        for value in target["source_taxon_id"].to_list()
        if str(value) not in completed
    ]
    if limit > 0:
        pending = pending[:limit]
    limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)
    semaphore = asyncio.Semaphore(max(1, workers))
    async with httpx.AsyncClient(
        base_url=CHECKLISTBANK_API,
        timeout=httpx.Timeout(timeout_seconds),
        limits=limits,
        headers={"User-Agent": "BioMiner/0.1 registry-name-harvester"},
    ) as client:
        for offset in range(0, len(pending), max(1, batch_size)):
            ids = pending[offset : offset + max(1, batch_size)]
            results = await asyncio.gather(
                *(
                    _fetch_names_for_taxon(
                        client,
                        semaphore,
                        taxon_id,
                        max_retries=max_retries,
                    )
                    for taxon_id in ids
                ),
                return_exceptions=True,
            )
            name_additions: list[dict[str, Any]] = []
            state_rows = state.to_dicts()
            state_by_id = {str(row["source_taxon_id"]): row for row in state_rows}
            now = datetime.now(UTC).isoformat()
            for taxon_id, result in zip(ids, results, strict=True):
                previous = state_by_id.get(taxon_id, {})
                attempts = int(previous.get("attempts") or 0) + 1
                if isinstance(result, BaseException):
                    state_by_id[taxon_id] = {
                        "source_taxon_id": taxon_id,
                        "status": "retryable",
                        "attempts": attempts,
                        "last_error": type(result).__name__,
                        "updated_at": now,
                    }
                else:
                    name_additions.extend(result)
                    state_by_id[taxon_id] = {
                        "source_taxon_id": taxon_id,
                        "status": "complete",
                        "attempts": attempts,
                        "last_error": "",
                        "updated_at": now,
                    }
            if name_additions:
                names = pl.concat(
                    [names, pl.DataFrame(name_additions, schema=_NAME_SCHEMA)],
                    how="vertical_relaxed",
                ).unique(
                    ["accepted_source_taxon_id", "display_name", "language", "name_class", "source"],
                    keep="first",
                    maintain_order=True,
                )
            state = pl.DataFrame(list(state_by_id.values()), schema=_NAME_STATE_SCHEMA)
            _write_source(nodes, names, source)
            write_parquet(state.sort("source_taxon_id"), source / SOURCE_NAME_STATE_FILE)
    return {
        "source_dir": str(source),
        "target_taxa": target.height,
        "completed_taxa": state.filter(pl.col("status") == "complete").height,
        "retryable_taxa": state.filter(pl.col("status") == "retryable").height,
        "name_rows": names.height,
        "synonym_rows": names.filter(pl.col("name_class") == "scientific_synonym").height,
        "vernacular_rows": names.filter(pl.col("name_class") == "vernacular").height,
    }


def col_xr_payload_from_parquet(source_dir: str | Path) -> dict[str, Any]:
    """Load a compiler payload from the durable Parquet source snapshot."""

    source = Path(source_dir)
    nodes = _read_typed(source / SOURCE_NODES_FILE, _NODE_SCHEMA)
    names = _read_typed(source / SOURCE_NAMES_FILE, _NAME_SCHEMA)
    if nodes.is_empty():
        raise ValueError(f"CoL XR Parquet source has no nodes: {source}")
    nodes = _attach_lineage(nodes)
    taxa_rows: list[dict[str, Any]] = []
    for row in nodes.iter_rows(named=True):
        rank = str(row["rank"])
        taxon_id = str(row["source_taxon_id"])
        key = f"col:{taxon_id}"
        taxa_rows.append(
            {
                "accepted_taxon_key": key,
                "scientific_name": row["scientific_name"],
                "rank": rank,
                "parent_key": f"col:{row['parent_source_taxon_id']}" if row["parent_source_taxon_id"] else "",
                "family_key": f"col:{row['family_source_taxon_id']}" if row["family_source_taxon_id"] else "",
                "family": row["family"],
                "genus_key": f"col:{row['genus_source_taxon_id']}" if row["genus_source_taxon_id"] else "",
                "genus": row["genus"],
                "species_key": key if rank == "SPECIES" else "",
                "species": row["scientific_name"] if rank == "SPECIES" else "",
                "status": "ACCEPTED",
                "source_taxon_id": taxon_id,
                "scientific_name_authorship": row["scientific_name_authorship"],
                "source_dataset_key": COL_XR_DATASET_KEY,
                "source_release": COL_XR_RELEASE,
            }
        )
    name_rows = [
        {
            "accepted_taxon_key": f"col:{row['accepted_source_taxon_id']}",
            "display_name": row["display_name"],
            "verbatim_name": row["display_name"],
            "language": row["language"],
            "name_class": row["name_class"],
            "source": row["source"],
            "source_record_id": row["source_record_id"],
            "source_taxon_id": row["accepted_source_taxon_id"],
            "trust_tier": row["trust_tier"],
            "precision_tier": "high",
            "confidence": "high",
            "review_state": row["review_state"],
            "enabled": row["enabled"],
        }
        for row in names.iter_rows(named=True)
    ]
    return {
        "source": "CoL XR",
        "source_version": COL_XR_RELEASE,
        "source_dataset_key": COL_XR_DATASET_KEY,
        "doi": COL_XR_DOI,
        "retrieved_at": _snapshot_retrieved_at(source),
        "source_url": f"{CHECKLISTBANK_API}/dataset/{COL_XR_DATASET_KEY}",
        "citation": f"Catalogue of Life {COL_XR_RELEASE}; DOI {COL_XR_DOI}",
        "taxa": taxa_rows,
        "names": name_rows,
    }


async def _fetch_children(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    taxon_id: str,
    *,
    max_retries: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = await _get_json(
            client,
            semaphore,
            f"/dataset/{COL_XR_DATASET_KEY}/tree/{taxon_id}/children",
            params={"limit": 1000, "offset": offset},
            max_retries=max_retries,
        )
        page = payload.get("result") or []
        for row in page:
            status = str(row.get("status") or "").casefold()
            rank = str(row.get("rank") or "").upper()
            if status not in {"accepted", "provisionally accepted"} or not rank:
                continue
            if rank in {"SUBSPECIES", "VARIETY", "FORM"}:
                continue
            rows.append(dict(row))
        offset += len(page)
        if not page or bool(payload.get("last")) or offset >= int(payload.get("total") or 0):
            return rows


async def _get_json(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    path: str,
    *,
    params: dict[str, Any],
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
                raise ValueError(f"ChecklistBank returned non-object JSON for {path}")
            return payload
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            await asyncio.sleep(min(30.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"ChecklistBank request failed after retries: {path}: {last_error}")


async def _fetch_names_for_taxon(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    taxon_id: str,
    *,
    max_retries: int,
) -> list[dict[str, Any]]:
    synonym_payload, vernacular_payload = await asyncio.gather(
        _get_any_json(
            client,
            semaphore,
            f"/dataset/{COL_XR_DATASET_KEY}/taxon/{taxon_id}/synonyms",
            max_retries=max_retries,
        ),
        _get_any_json(
            client,
            semaphore,
            f"/dataset/{COL_XR_DATASET_KEY}/taxon/{taxon_id}/vernacular",
            max_retries=max_retries,
        ),
    )
    rows: list[dict[str, Any]] = []
    for synonym in _json_rows(synonym_payload):
        name = synonym.get("name")
        if isinstance(name, dict):
            display_name = str(name.get("scientificName") or "")
        else:
            display_name = str(synonym.get("scientificName") or name or "")
        if display_name:
            rows.append(
                {
                    "accepted_source_taxon_id": taxon_id,
                    "display_name": display_name,
                    "language": "la",
                    "name_class": "scientific_synonym",
                    "source": "CoL XR",
                    "source_record_id": str(synonym.get("id") or f"{taxon_id}:{display_name}"),
                    "trust_tier": "T1",
                    "review_state": "source_accepted",
                    "enabled": True,
                }
            )
    for vernacular in _json_rows(vernacular_payload):
        display_name = str(vernacular.get("name") or "")
        if display_name:
            rows.append(
                {
                    "accepted_source_taxon_id": taxon_id,
                    "display_name": display_name,
                    "language": str(vernacular.get("language") or ""),
                    "name_class": "vernacular",
                    "source": "CoL XR",
                    "source_record_id": str(vernacular.get("id") or f"{taxon_id}:{display_name}"),
                    "trust_tier": "T1",
                    "review_state": "source_accepted",
                    "enabled": True,
                }
            )
    return rows


async def _get_any_json(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    path: str,
    *,
    max_retries: int,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                response = await client.get(path)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            await asyncio.sleep(min(30.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"ChecklistBank request failed after retries: {path}: {last_error}")


def _json_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        # Synonyms are grouped by nomenclatural relationship. Read only the
        # flat groups; *Groups repeats the same usages in nested arrays.
        for key in (
            "result",
            "results",
            "synonyms",
            "heterotypic",
            "homotypic",
            "proparte",
            "misapplied",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def _seed_nodes(included_families: Iterable[str]) -> pl.DataFrame:
    family_ids = {
        "Hedylidae": "62533",
        "Hesperiidae": "6254D",
        "Lycaenidae": "C98",
        "Nymphalidae": "DGC",
        "Papilionidae": "DX5",
        "Pieridae": "625MK",
        "Riodinidae": "FSJ",
    }
    top = [
        ("N", "", "Animalia", "", "KINGDOM", 1),
        ("RT", "N", "Arthropoda", "", "PHYLUM", 1),
        ("H6", "RT", "Insecta", "", "CLASS", 1),
        ("B6L67", "H6", "Lepidoptera", "Linnaeus, 1758", "ORDER", 1),
        (COL_XR_ROOT_ID, "B6L67", "Papilionoidea", "", "SUPERFAMILY", 7),
    ]
    rows = [
        {
            "id": taxon_id,
            "parentId": parent_id,
            "name": name,
            "authorship": authorship,
            "rank": rank,
            "status": "accepted",
            "childCount": child_count,
            "expanded": True,
        }
        for taxon_id, parent_id, name, authorship, rank, child_count in top
    ]
    for family in included_families:
        if family not in family_ids:
            raise ValueError(f"CoL XR family ID is not configured: {family}")
        rows.append(
            {
                "id": family_ids[family],
                "parentId": COL_XR_ROOT_ID,
                "name": family,
                "authorship": "",
                "rank": "FAMILY",
                "status": "accepted",
                "childCount": 1,
                "expanded": False,
            }
        )
    return _attach_lineage(_node_frame(rows))


def _node_frame(rows: Iterable[dict[str, Any]]) -> pl.DataFrame:
    normalized = [
        {
            "source_taxon_id": str(row.get("source_taxon_id") or row.get("id") or ""),
            "parent_source_taxon_id": str(row.get("parent_source_taxon_id") or row.get("parentId") or ""),
            "scientific_name": str(row.get("scientific_name") or row.get("name") or ""),
            "scientific_name_authorship": str(row.get("scientific_name_authorship") or row.get("authorship") or ""),
            "rank": str(row.get("rank") or "").upper(),
            "source_status": str(row.get("source_status") or row.get("status") or ""),
            "child_count": int(row.get("child_count") or row.get("childCount") or 0),
            "expanded": bool(row.get("expanded", False)),
            "family_source_taxon_id": str(row.get("family_source_taxon_id") or ""),
            "family": str(row.get("family") or ""),
            "genus_source_taxon_id": str(row.get("genus_source_taxon_id") or ""),
            "genus": str(row.get("genus") or ""),
        }
        for row in rows
        if str(row.get("source_taxon_id") or row.get("id") or "")
    ]
    return pl.DataFrame(normalized, schema=_NODE_SCHEMA)


def _attach_lineage(nodes: pl.DataFrame) -> pl.DataFrame:
    if nodes.is_empty():
        return nodes
    rows = {str(row["source_taxon_id"]): dict(row) for row in nodes.iter_rows(named=True)}
    for row in rows.values():
        family_id = ""
        family = ""
        genus_id = ""
        genus = ""
        current = row
        visited: set[str] = set()
        while current and str(current["source_taxon_id"]) not in visited:
            visited.add(str(current["source_taxon_id"]))
            rank = str(current["rank"])
            if rank == "FAMILY" and not family_id:
                family_id = str(current["source_taxon_id"])
                family = str(current["scientific_name"])
            if rank == "GENUS" and not genus_id:
                genus_id = str(current["source_taxon_id"])
                genus = str(current["scientific_name"])
            current = rows.get(str(current["parent_source_taxon_id"]))
        row["family_source_taxon_id"] = family_id
        row["family"] = family
        row["genus_source_taxon_id"] = genus_id
        row["genus"] = genus
    genus_lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows.values():
        if str(row["rank"]) == "GENUS":
            genus_lookup.setdefault(
                (str(row["family_source_taxon_id"]), str(row["scientific_name"])), []
            ).append(row)
    for row in rows.values():
        if str(row["rank"]) != "SPECIES" or str(row["genus_source_taxon_id"]):
            continue
        genus_name = str(row["scientific_name"]).partition(" ")[0]
        matches = genus_lookup.get((str(row["family_source_taxon_id"]), genus_name), [])
        if len(matches) == 1:
            row["genus_source_taxon_id"] = str(matches[0]["source_taxon_id"])
            row["genus"] = genus_name
    return pl.DataFrame(list(rows.values()), schema=_NODE_SCHEMA).sort(
        ["rank", "scientific_name", "source_taxon_id"]
    )


def _accepted_name_rows(nodes: pl.DataFrame) -> pl.DataFrame:
    rows = [
        {
            "accepted_source_taxon_id": str(row["source_taxon_id"]),
            "display_name": str(row["scientific_name"]),
            "language": "la",
            "name_class": "accepted_scientific",
            "source": "CoL XR",
            "source_record_id": str(row["source_taxon_id"]),
            "trust_tier": "T1",
            "review_state": "source_accepted",
            "enabled": True,
        }
        for row in nodes.iter_rows(named=True)
        if str(row["scientific_name"])
    ]
    return pl.DataFrame(rows, schema=_NAME_SCHEMA)


def _read_typed(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not path.is_file():
        return pl.DataFrame(schema=schema)
    return pl.read_parquet(path).cast(schema)


def _write_source(nodes: pl.DataFrame, names: pl.DataFrame, output: Path) -> None:
    write_parquet(nodes.sort(["rank", "scientific_name", "source_taxon_id"]), output / SOURCE_NODES_FILE)
    write_parquet(
        names.sort(["accepted_source_taxon_id", "name_class", "language", "display_name"]),
        output / SOURCE_NAMES_FILE,
    )


def _snapshot_retrieved_at(source: Path) -> str:
    path = source / SOURCE_SNAPSHOTS_FILE
    if not path.is_file():
        return datetime.now(UTC).isoformat()
    rows = pl.read_parquet(path)
    return str(rows["retrieved_at"][0]) if rows.height else datetime.now(UTC).isoformat()


__all__ = [
    "SOURCE_NAMES_FILE",
    "SOURCE_NAME_STATE_FILE",
    "SOURCE_NODES_FILE",
    "SOURCE_SNAPSHOTS_FILE",
    "col_xr_payload_from_parquet",
    "harvest_col_xr_taxonomy",
    "harvest_col_xr_names",
]
