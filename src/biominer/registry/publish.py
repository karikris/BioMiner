from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import polars as pl

from biominer.registry.geographic_spread import (
    GEOGRAPHIC_OCCURRENCE_EVIDENCE_FILE,
    GEOGRAPHIC_SPREAD_MANIFEST_FILE,
    TAXON_GEOGRAPHIC_SPREAD_FILE,
    geographic_occurrence_evidence_schema,
    geographic_spread_schema,
)
from biominer.registry.geographic_summary import (
    GEOGRAPHIC_QA_FINDINGS_FILE,
    GEOGRAPHIC_SUMMARY_MANIFEST_FILE,
    TAXON_GEOGRAPHIC_SUMMARY_FILE,
    geographic_qa_schema,
    geographic_spread_fingerprint,
    geographic_summary_schema,
)
from biominer.storage.parquet import write_parquet


PUBLISHED_REGISTRY_ARTIFACTS = (
    "taxa.parquet",
    "species_paths.parquet",
    "names.parquet",
    "flickr_query_definitions.parquet",
    "source_snapshots.parquet",
    "qa_findings.parquet",
    TAXON_GEOGRAPHIC_SPREAD_FILE,
    TAXON_GEOGRAPHIC_SUMMARY_FILE,
    "manifest.json",
)
GEOGRAPHIC_STAGING_ARTIFACTS = (
    GEOGRAPHIC_OCCURRENCE_EVIDENCE_FILE,
    GEOGRAPHIC_QA_FINDINGS_FILE,
    GEOGRAPHIC_SPREAD_MANIFEST_FILE,
    GEOGRAPHIC_SUMMARY_MANIFEST_FILE,
)
REGISTRY_PUBLICATION_MANIFEST_SCHEMA_VERSION = "registry-publication-v2.0.0"

_SOURCE_SNAPSHOT_SCHEMA = {
    "source": pl.String,
    "source_version": pl.String,
    "retrieved_at": pl.String,
    "source_path": pl.String,
    "source_response_hash": pl.String,
    "licence": pl.String,
    "source_url": pl.String,
    "citation": pl.String,
}


def publish_registry(
    registry_dir: str | Path,
    *,
    output_dir: str | Path = "data/registry/current",
    replace_existing: bool = False,
) -> dict[str, Any]:
    source = Path(registry_dir)
    target = Path(output_dir)
    required = (*PUBLISHED_REGISTRY_ARTIFACTS, *GEOGRAPHIC_STAGING_ARTIFACTS)
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError("registry publication is missing artifacts: " + ", ".join(missing))
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("qa_status") != "passed" or int(manifest.get("qa_fatal_count") or 0):
        raise ValueError("registry publication requires a fatal-QA-clean manifest")
    taxa = pl.read_parquet(source / "taxa.parquet")
    _validate_species_paths(pl.read_parquet(source / "species_paths.parquet"), taxa)
    _validate_keywords(
        pl.read_parquet(source / "names.parquet"),
        pl.read_parquet(source / "flickr_query_definitions.parquet"),
    )
    spread = pl.read_parquet(source / TAXON_GEOGRAPHIC_SPREAD_FILE)
    evidence = pl.read_parquet(source / GEOGRAPHIC_OCCURRENCE_EVIDENCE_FILE)
    summary = pl.read_parquet(source / TAXON_GEOGRAPHIC_SUMMARY_FILE)
    geographic_qa = pl.read_parquet(source / GEOGRAPHIC_QA_FINDINGS_FILE)
    _validate_geographic_artifacts(
        taxa=taxa,
        spread=spread,
        evidence=evidence,
        summary=summary,
        geographic_qa=geographic_qa,
        registry_version=str(manifest.get("registry_version") or ""),
    )
    _validate_geographic_manifests(source, geographic_qa=geographic_qa)
    geographic_fatal_count = geographic_qa.filter(pl.col("severity") == "fatal").height
    if geographic_fatal_count:
        raise ValueError("registry publication is blocked by fatal geographic QA")
    qa = _merged_qa(
        pl.read_parquet(source / "qa_findings.parquet"),
        geographic_qa,
    )
    source_snapshots, geographic_snapshot_count = _merged_source_snapshots(
        base=pl.read_parquet(source / "source_snapshots.parquet"),
        spread=spread,
        evidence=evidence,
        spread_manifest=_read_json(source / GEOGRAPHIC_SPREAD_MANIFEST_FILE),
        evidence_path=source / GEOGRAPHIC_OCCURRENCE_EVIDENCE_FILE,
    )
    if target.exists() and not replace_existing:
        raise FileExistsError(
            f"registry publication target exists: {target}; use --replace-existing"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
        staged = Path(temporary) / target.name
        staged.mkdir()
        for name in (
            "taxa.parquet",
            "species_paths.parquet",
            "names.parquet",
            "flickr_query_definitions.parquet",
            TAXON_GEOGRAPHIC_SPREAD_FILE,
            TAXON_GEOGRAPHIC_SUMMARY_FILE,
        ):
            shutil.copy2(source / name, staged / name)
        write_parquet(source_snapshots, staged / "source_snapshots.parquet")
        write_parquet(qa, staged / "qa_findings.parquet")
        published_manifest = _published_manifest(
            base=manifest,
            staged=staged,
            spread=spread,
            summary=summary,
            qa=qa,
            geographic_qa=geographic_qa,
            source_snapshots=source_snapshots,
            geographic_snapshot_count=geographic_snapshot_count,
        )
        # The manifest remains the final staged object for local and cloud adapters.
        (staged / "manifest.json").write_text(
            json.dumps(published_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _verify_inventory(staged, published_manifest["artifact_inventory"])
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


def _validate_geographic_artifacts(
    *,
    taxa: pl.DataFrame,
    spread: pl.DataFrame,
    evidence: pl.DataFrame,
    summary: pl.DataFrame,
    geographic_qa: pl.DataFrame,
    registry_version: str,
) -> None:
    expected_schemas = (
        (spread, geographic_spread_schema(), "taxon geographic spread"),
        (evidence, geographic_occurrence_evidence_schema(), "geographic occurrence evidence"),
        (summary, geographic_summary_schema(), "taxon geographic summary"),
        (geographic_qa, geographic_qa_schema(), "geographic QA"),
    )
    for frame, expected, name in expected_schemas:
        if frame.schema != expected:
            raise ValueError(f"{name} schema mismatch")
    accepted_species = set(
        taxa.filter(
            (pl.col("rank") == "SPECIES") & (pl.col("taxonomic_status") == "ACCEPTED")
        )["accepted_taxon_key"].to_list()
    )
    summary_species = set(summary["accepted_taxon_key"].to_list())
    if summary_species != accepted_species:
        raise ValueError("taxon geographic summary must cover every accepted species")
    if summary["accepted_taxon_key"].n_unique() != summary.height:
        raise ValueError("taxon geographic summary must contain one current row per species")
    spread_species = set(spread["accepted_taxon_key"].to_list())
    if not spread_species <= accepted_species:
        raise ValueError("taxon geographic spread contains taxa outside the accepted registry")
    for row in summary.select(["accepted_taxon_key", "spread_fingerprint"]).to_dicts():
        taxon_spread = spread.filter(
            pl.col("accepted_taxon_key") == row["accepted_taxon_key"]
        )
        if row["spread_fingerprint"] != geographic_spread_fingerprint(taxon_spread):
            raise ValueError(
                "taxon geographic summary spread fingerprint mismatch: "
                + str(row["accepted_taxon_key"])
            )
    for frame, name in ((spread, "spread"), (evidence, "evidence"), (summary, "summary")):
        if not frame.is_empty() and set(frame["registry_version"].to_list()) != {
            registry_version
        }:
            raise ValueError(f"geographic {name} registry_version mismatch")


def _validate_geographic_manifests(
    source: Path,
    *,
    geographic_qa: pl.DataFrame,
) -> None:
    spread_manifest = _read_json(source / GEOGRAPHIC_SPREAD_MANIFEST_FILE)
    summary_manifest = _read_json(source / GEOGRAPHIC_SUMMARY_MANIFEST_FILE)
    if spread_manifest.get("status") != "complete":
        raise ValueError("geographic spread manifest is not complete")
    if summary_manifest.get("status") != "complete":
        raise ValueError("geographic summary manifest is not complete")
    expected_qa_status = (
        "failed"
        if geographic_qa.filter(pl.col("severity") == "fatal").height
        else "passed"
    )
    if summary_manifest.get("qa_status") != expected_qa_status:
        raise ValueError("geographic summary manifest QA status mismatch")
    _validate_manifest_files(
        source,
        spread_manifest,
        required={TAXON_GEOGRAPHIC_SPREAD_FILE, GEOGRAPHIC_OCCURRENCE_EVIDENCE_FILE},
    )
    _validate_manifest_files(
        source,
        summary_manifest,
        required={TAXON_GEOGRAPHIC_SUMMARY_FILE, GEOGRAPHIC_QA_FINDINGS_FILE},
    )


def _validate_manifest_files(
    source: Path,
    manifest: dict[str, object],
    *,
    required: set[str],
) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("geographic build manifest files inventory is missing")
    entries = {
        str(entry.get("file") or ""): entry
        for entry in files.values()
        if isinstance(entry, dict)
    }
    missing = sorted(required - set(entries))
    if missing:
        raise ValueError(
            "geographic build manifest files inventory is incomplete: " + ", ".join(missing)
        )
    for filename in sorted(required):
        path = source / filename
        expected = entries[filename]
        actual = _artifact_entry(path)
        for field_name in ("row_count", "byte_count", "sha256"):
            if expected.get(field_name) != actual[field_name]:
                raise ValueError(
                    f"geographic build artifact {field_name} mismatch: {filename}"
                )


def _merged_qa(base: pl.DataFrame, geographic: pl.DataFrame) -> pl.DataFrame:
    if base.schema != geographic_qa_schema():
        raise ValueError("registry QA schema mismatch")
    return (
        pl.concat([base, geographic], how="vertical")
        .unique(subset=["severity", "code", "subject"], maintain_order=False)
        .sort(["severity", "code", "subject"])
    )


def _merged_source_snapshots(
    *,
    base: pl.DataFrame,
    spread: pl.DataFrame,
    evidence: pl.DataFrame,
    spread_manifest: dict[str, object],
    evidence_path: Path,
) -> tuple[pl.DataFrame, int]:
    if base.schema != _SOURCE_SNAPSHOT_SCHEMA:
        raise ValueError("registry source snapshot schema mismatch")
    geographic = _geographic_source_snapshots(
        spread=spread,
        evidence=evidence,
        spread_manifest=spread_manifest,
        evidence_path=evidence_path,
    )
    merged = (
        pl.concat([base, geographic], how="vertical")
        .unique(maintain_order=False)
        .sort(["source", "source_version", "source_path", "source_response_hash"])
    )
    return merged, geographic.height


def _geographic_source_snapshots(
    *,
    spread: pl.DataFrame,
    evidence: pl.DataFrame,
    spread_manifest: dict[str, object],
    evidence_path: Path,
) -> pl.DataFrame:
    source_rows: list[dict[str, object]] = []
    for frame in (spread, evidence):
        if frame.is_empty():
            continue
        source_rows.extend(
            frame.select(
                [
                    "source",
                    "source_snapshot_version",
                    "retrieved_at",
                    "source_dataset_citation",
                ]
            )
            .unique(maintain_order=False)
            .to_dicts()
        )
    if not source_rows:
        source_rows.append(
            {
                "source": spread_manifest.get("source"),
                "source_snapshot_version": spread_manifest.get("source_snapshot_version"),
                "retrieved_at": spread_manifest.get("retrieved_at"),
                "source_dataset_citation": None,
            }
        )
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in source_rows:
        source = str(row.get("source") or "").strip()
        version = str(row.get("source_snapshot_version") or "").strip()
        if not source or not version:
            raise ValueError("geographic source snapshot identity must be nonblank")
        grouped.setdefault((source, version), []).append(row)
    normalized: list[dict[str, object]] = []
    evidence_hash = _sha256_file(evidence_path)
    for (source, version), rows in sorted(grouped.items()):
        retrieved_values = {
            _timestamp_text(row.get("retrieved_at")) for row in rows if row.get("retrieved_at")
        }
        if len(retrieved_values) != 1:
            raise ValueError(f"geographic source snapshot retrieval time is ambiguous: {version}")
        citations = sorted(
            {
                str(row.get("source_dataset_citation") or "").strip()
                for row in rows
                if str(row.get("source_dataset_citation") or "").strip()
            }
        )
        normalized.append(
            {
                "source": source,
                "source_version": version,
                "retrieved_at": next(iter(retrieved_values)),
                "source_path": GEOGRAPHIC_OCCURRENCE_EVIDENCE_FILE,
                "source_response_hash": evidence_hash,
                "licence": "",
                "source_url": "",
                "citation": citations[0] if len(citations) == 1 else "",
            }
        )
    return pl.DataFrame(normalized, schema=_SOURCE_SNAPSHOT_SCHEMA)


def _published_manifest(
    *,
    base: dict[str, object],
    staged: Path,
    spread: pl.DataFrame,
    summary: pl.DataFrame,
    qa: pl.DataFrame,
    geographic_qa: pl.DataFrame,
    source_snapshots: pl.DataFrame,
    geographic_snapshot_count: int,
) -> dict[str, object]:
    fatal_count = qa.filter(pl.col("severity") == "fatal").height
    warning_count = qa.filter(pl.col("severity") == "warning").height
    geographic_fatal_count = geographic_qa.filter(pl.col("severity") == "fatal").height
    geographic_warning_count = geographic_qa.filter(pl.col("severity") == "warning").height
    inventory = {
        filename: _artifact_entry(staged / filename)
        for filename in PUBLISHED_REGISTRY_ARTIFACTS
        if filename != "manifest.json"
    }
    return {
        **base,
        "publication_manifest_schema_version": REGISTRY_PUBLICATION_MANIFEST_SCHEMA_VERSION,
        "published_artifacts": list(PUBLISHED_REGISTRY_ARTIFACTS),
        "manifest_written_last": True,
        "artifact_inventory": inventory,
        "source_snapshot_rows": source_snapshots.height,
        "geographic_source_snapshot_rows": geographic_snapshot_count,
        "geographic_spread_rows": spread.height,
        "geographic_summary_rows": summary.height,
        "geographic_data_deficient_species": summary.filter(pl.col("data_deficient")).height,
        "geographic_qa_fatal_count": geographic_fatal_count,
        "geographic_qa_warning_count": geographic_warning_count,
        "qa_finding_rows": qa.height,
        "qa_fatal_count": fatal_count,
        "qa_warning_count": warning_count,
        "qa_status": "failed" if fatal_count else "passed",
        "geographic_absence_semantics": "unknown_not_negative",
    }


def _artifact_entry(path: Path) -> dict[str, object]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    try:
        row_count = int(parquet.metadata.num_rows)
    finally:
        parquet.close()
    return {
        "byte_count": path.stat().st_size,
        "row_count": row_count,
        "sha256": _sha256_file(path),
    }


def _verify_inventory(staged: Path, inventory: object) -> None:
    if not isinstance(inventory, dict):
        raise ValueError("registry artifact inventory must be an object")
    for filename, value in inventory.items():
        if not isinstance(value, dict):
            raise ValueError(f"invalid registry artifact inventory entry: {filename}")
        path = staged / str(filename)
        if not path.is_file():
            raise ValueError(f"registry artifact disappeared before promotion: {filename}")
        if path.stat().st_size != value.get("byte_count"):
            raise ValueError(f"registry artifact byte-count mismatch: {filename}")
        if _sha256_file(path) != value.get("sha256"):
            raise ValueError(f"registry artifact checksum mismatch: {filename}")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _timestamp_text(value: object) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("geographic source retrieval time must include a timezone")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("geographic source retrieval time must include a timezone")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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
