from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import polars as pl


PUBLISHED_REGISTRY_ARTIFACTS = (
    "taxa.parquet",
    "species_paths.parquet",
    "names.parquet",
    "flickr_query_definitions.parquet",
    "source_snapshots.parquet",
    "qa_findings.parquet",
    "manifest.json",
)


def publish_registry(
    registry_dir: str | Path,
    *,
    output_dir: str | Path = "data/registry/current",
    replace_existing: bool = False,
) -> dict[str, Any]:
    source = Path(registry_dir)
    target = Path(output_dir)
    missing = [name for name in PUBLISHED_REGISTRY_ARTIFACTS if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError("registry publication is missing artifacts: " + ", ".join(missing))
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("qa_status") != "passed" or int(manifest.get("qa_fatal_count") or 0):
        raise ValueError("registry publication requires a fatal-QA-clean manifest")
    _validate_species_paths(pl.read_parquet(source / "species_paths.parquet"), pl.read_parquet(source / "taxa.parquet"))
    _validate_keywords(
        pl.read_parquet(source / "names.parquet"),
        pl.read_parquet(source / "flickr_query_definitions.parquet"),
    )
    if target.exists() and not replace_existing:
        raise FileExistsError(f"registry publication target exists: {target}; use --replace-existing")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
        staged = Path(temporary) / target.name
        staged.mkdir()
        for name in PUBLISHED_REGISTRY_ARTIFACTS[:-1]:
            shutil.copy2(source / name, staged / name)
        # Manifest is deliberately staged last and therefore remains the final
        # object for cloud adapters built on this publication contract.
        shutil.copy2(source / "manifest.json", staged / "manifest.json")
        if target.exists():
            if target.is_symlink() or target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)
        os.replace(staged, target)
    return {
        "status": "published",
        "registry_dir": str(source),
        "output_dir": str(target),
        "registry_version": str(manifest.get("registry_version") or ""),
        "artifacts": list(PUBLISHED_REGISTRY_ARTIFACTS),
        "manifest_written_last": True,
    }


def _validate_species_paths(paths: pl.DataFrame, taxa: pl.DataFrame) -> None:
    accepted_species = taxa.filter(
        (pl.col("rank") == "SPECIES") & (pl.col("taxonomic_status") == "ACCEPTED")
    )["accepted_taxon_key"].to_list()
    enabled = paths.filter(pl.col("enabled"))
    if enabled["accepted_taxon_key"].n_unique() != enabled.height:
        raise ValueError("species_paths must contain exactly one enabled path per accepted taxon")
    if set(enabled["accepted_taxon_key"].to_list()) != set(accepted_species):
        raise ValueError("species_paths do not cover every accepted species exactly once")
    path_columns = [
        f"{rank}_node_id"
        for rank in ("kingdom", "phylum", "class", "order", "family", "genus", "species")
    ]
    if enabled.select(pl.any_horizontal(*(pl.col(column) == "" for column in path_columns))).to_series().any():
        raise ValueError("species_paths contain a structurally incomplete enabled path")


def _validate_keywords(names: pl.DataFrame, queries: pl.DataFrame) -> None:
    canonical = names.filter(pl.col("is_canonical_keyword"))
    if canonical["normalized_match_key"].n_unique() != canonical.height:
        raise ValueError("names contain duplicate canonical normalized terms")
    if queries["logical_query_id"].n_unique() != queries.height:
        raise ValueError("flickr query definitions contain duplicate logical queries")
    duplicate_term_fields = queries.group_by(["normalized_match_key", "search_field"]).len().filter(pl.col("len") > 1)
    if not duplicate_term_fields.is_empty():
        raise ValueError("flickr query definitions duplicate normalized term/search-field requests")
