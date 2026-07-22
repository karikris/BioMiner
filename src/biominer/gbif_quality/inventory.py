from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint


INVENTORY_SCHEMA_VERSION = "biominer-gbif-media-source-inventory/v1"
REQUIRED_DWCA_MEMBERS = ("occurrence.txt", "multimedia.txt", "verbatim.txt")
_DOWNLOAD_KEY = re.compile(r"GBIF Occurrence Download\s+(?P<key>[^\s<]+)")

INVENTORY_SCHEMA = pa.schema(
    [
        ("source_snapshot_id", pa.string()),
        ("artifact_role", pa.string()),
        ("path", pa.string()),
        ("member", pa.string()),
        ("physical_bytes", pa.int64()),
        ("sha256", pa.string()),
        ("expected_sha256", pa.string()),
        ("checksum_status", pa.string()),
        ("row_count", pa.int64()),
        ("expected_row_count", pa.int64()),
        ("row_count_status", pa.string()),
        ("column_count", pa.int32()),
        ("expected_column_count", pa.int32()),
        ("column_count_status", pa.string()),
        ("row_group_count", pa.int32()),
        ("row_groups_complete", pa.bool_()),
        ("schema_fingerprint", pa.string()),
        ("manifest_path", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class SourceInventoryConfig:
    repository_root: Path
    archive: Path
    dwca_manifest: Path
    occurrence_parquet: Path
    multimedia_parquet: Path
    verbatim_parquet: Path
    joined_parquet: Path
    joined_manifest: Path
    v3_parquet: Path
    v3_manifest: Path
    prior_intake_manifest: Path | None = None

    def resolved(self) -> SourceInventoryConfig:
        root = self.repository_root.resolve()
        return SourceInventoryConfig(
            repository_root=root,
            archive=_resolve(root, self.archive),
            dwca_manifest=_resolve(root, self.dwca_manifest),
            occurrence_parquet=_resolve(root, self.occurrence_parquet),
            multimedia_parquet=_resolve(root, self.multimedia_parquet),
            verbatim_parquet=_resolve(root, self.verbatim_parquet),
            joined_parquet=_resolve(root, self.joined_parquet),
            joined_manifest=_resolve(root, self.joined_manifest),
            v3_parquet=_resolve(root, self.v3_parquet),
            v3_manifest=_resolve(root, self.v3_manifest),
            prior_intake_manifest=(
                _resolve(root, self.prior_intake_manifest)
                if self.prior_intake_manifest is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceInventory:
    schema_version: str
    source_snapshot_id: str
    source_download_key: str
    source_package_id: str | None
    source_title: str
    source_publication_date: str | None
    archive_members: tuple[dict[str, Any], ...]
    artifacts: tuple[dict[str, Any], ...]
    validation: dict[str, bool]

    def artifact_table(self) -> pa.Table:
        return pa.Table.from_pylist(list(self.artifacts), schema=INVENTORY_SCHEMA)

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_snapshot_id": self.source_snapshot_id,
            "source_download_key": self.source_download_key,
            "source_package_id": self.source_package_id,
            "source_title": self.source_title,
            "source_publication_date": self.source_publication_date,
            "archive_members": list(self.archive_members),
            "artifacts": list(self.artifacts),
            "validation": self.validation,
        }


def build_source_inventory(config: SourceInventoryConfig) -> SourceInventory:
    """Recalculate the source/archive inventory and fail closed on drift."""

    cfg = config.resolved()
    for path in (
        cfg.archive,
        cfg.dwca_manifest,
        cfg.occurrence_parquet,
        cfg.multimedia_parquet,
        cfg.verbatim_parquet,
        cfg.joined_parquet,
        cfg.joined_manifest,
        cfg.v3_parquet,
        cfg.v3_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    dwca = _read_json(cfg.dwca_manifest)
    joined = _read_json(cfg.joined_manifest)
    v3 = _read_json(cfg.v3_manifest)
    prior = (
        _read_json(cfg.prior_intake_manifest)
        if cfg.prior_intake_manifest is not None
        else None
    )

    archive_sha = _sha256(cfg.archive)
    expected_archive_sha = _strip_sha256(dwca["source"]["archive_sha256"])
    if archive_sha != expected_archive_sha:
        raise ValueError(
            "archive checksum mismatch: "
            f"expected {expected_archive_sha}, found {archive_sha}"
        )
    eml = _read_archive_metadata(cfg.archive)
    snapshot_id = canonical_semantic_fingerprint(
        {
            "contract": INVENTORY_SCHEMA_VERSION,
            "archive_sha256": archive_sha,
            "download_key": eml["download_key"],
            "package_id": eml["package_id"],
        }
    )
    archive_members = _archive_member_inventory(cfg.archive)
    required_members_valid = all(
        any(item["member"] == required for item in archive_members)
        for required in REQUIRED_DWCA_MEMBERS
    )

    expected_by_path = _expected_parquets(
        cfg=cfg,
        dwca=dwca,
        joined=joined,
        v3=v3,
        prior=prior,
    )
    artifacts: list[dict[str, Any]] = [
        {
            "source_snapshot_id": snapshot_id,
            "artifact_role": "dwca_archive",
            "path": _display_path(cfg.repository_root, cfg.archive),
            "member": None,
            "physical_bytes": cfg.archive.stat().st_size,
            "sha256": archive_sha,
            "expected_sha256": expected_archive_sha,
            "checksum_status": "PASS",
            "row_count": None,
            "expected_row_count": None,
            "row_count_status": "NOT_APPLICABLE",
            "column_count": None,
            "expected_column_count": None,
            "column_count_status": "NOT_APPLICABLE",
            "row_group_count": None,
            "row_groups_complete": None,
            "schema_fingerprint": None,
            "manifest_path": _display_path(cfg.repository_root, cfg.dwca_manifest),
        }
    ]
    roles = (
        ("occurrence_core", cfg.occurrence_parquet),
        ("multimedia_extension", cfg.multimedia_parquet),
        ("verbatim_extension", cfg.verbatim_parquet),
        ("occurrence_multimedia_join", cfg.joined_parquet),
        ("rights_filtered_v3", cfg.v3_parquet),
    )
    for role, path in roles:
        artifacts.append(
            _inventory_parquet(
                root=cfg.repository_root,
                snapshot_id=snapshot_id,
                role=role,
                path=path,
                expected=expected_by_path[path],
            )
        )
    validation = {
        "archive_checksum_matches": archive_sha == expected_archive_sha,
        "required_archive_members_present": required_members_valid,
        "all_artifact_checksums_match": all(
            row["checksum_status"] in {"PASS", "NOT_RECORDED"}
            for row in artifacts
        ),
        "all_recorded_row_counts_match": all(
            row["row_count_status"] in {"PASS", "NOT_APPLICABLE"}
            for row in artifacts
        ),
        "all_recorded_column_counts_match": all(
            row["column_count_status"] in {"PASS", "NOT_APPLICABLE"}
            for row in artifacts
        ),
        "all_parquet_row_groups_complete": all(
            row["row_groups_complete"] is not False for row in artifacts
        ),
    }
    if not all(validation.values()):
        raise ValueError(f"source inventory validation failed: {validation}")
    return SourceInventory(
        schema_version=INVENTORY_SCHEMA_VERSION,
        source_snapshot_id=snapshot_id,
        source_download_key=eml["download_key"],
        source_package_id=eml["package_id"],
        source_title=eml["title"],
        source_publication_date=eml["publication_date"],
        archive_members=tuple(archive_members),
        artifacts=tuple(artifacts),
        validation=validation,
    )


def _inventory_parquet(
    *,
    root: Path,
    snapshot_id: str,
    role: str,
    path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    row_groups = [
        metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)
    ]
    checksum = _sha256(path)
    expected_checksum = _strip_sha256(expected.get("sha256"))
    expected_rows = _optional_int(expected.get("row_count"))
    expected_columns = _optional_int(expected.get("column_count"))
    return {
        "source_snapshot_id": snapshot_id,
        "artifact_role": role,
        "path": _display_path(root, path),
        "member": expected.get("member"),
        "physical_bytes": path.stat().st_size,
        "sha256": checksum,
        "expected_sha256": expected_checksum,
        "checksum_status": (
            "NOT_RECORDED"
            if expected_checksum is None
            else "PASS" if checksum == expected_checksum else "FAIL"
        ),
        "row_count": metadata.num_rows,
        "expected_row_count": expected_rows,
        "row_count_status": (
            "NOT_APPLICABLE"
            if expected_rows is None
            else "PASS" if metadata.num_rows == expected_rows else "FAIL"
        ),
        "column_count": len(parquet.schema_arrow),
        "expected_column_count": expected_columns,
        "column_count_status": (
            "NOT_APPLICABLE"
            if expected_columns is None
            else "PASS" if len(parquet.schema_arrow) == expected_columns else "FAIL"
        ),
        "row_group_count": metadata.num_row_groups,
        "row_groups_complete": bool(row_groups)
        and all(value > 0 for value in row_groups)
        and sum(row_groups) == metadata.num_rows,
        "schema_fingerprint": canonical_semantic_fingerprint(
            {
                "fields": [
                    {
                        "name": field.name,
                        "type": str(field.type),
                        "nullable": field.nullable,
                    }
                    for field in parquet.schema_arrow
                ]
            }
        ),
        "manifest_path": _display_path(root, expected["manifest"]),
    }


def _expected_parquets(
    *,
    cfg: SourceInventoryConfig,
    dwca: dict[str, Any],
    joined: dict[str, Any],
    v3: dict[str, Any],
    prior: dict[str, Any] | None,
) -> dict[Path, dict[str, Any]]:
    result: dict[Path, dict[str, Any]] = {}
    raw_roles = {
        "occurrence.txt": cfg.occurrence_parquet,
        "multimedia.txt": cfg.multimedia_parquet,
        "verbatim.txt": cfg.verbatim_parquet,
    }
    prior_parquets = {
        _resolve(cfg.repository_root, Path(item["path"])): item
        for item in (prior or {}).get("inputs", {}).get("parquets", [])
    }
    for item in dwca["outputs"]:
        member = str(item["member"])
        path = raw_roles[member]
        prior_item = prior_parquets.get(path, {})
        result[path] = {
            "member": member,
            "row_count": item.get("row_count"),
            "column_count": item.get("column_count"),
            "sha256": prior_item.get("sha256"),
            "manifest": cfg.dwca_manifest,
        }
    joined_output = joined["output"]
    result[cfg.joined_parquet] = {
        "member": None,
        "row_count": joined_output.get("row_count"),
        "column_count": joined_output.get("column_count"),
        "sha256": joined_output.get("physical_sha256"),
        "manifest": cfg.joined_manifest,
    }
    v3_input = v3["input"]
    result[cfg.v3_parquet] = {
        "member": None,
        "row_count": v3_input.get("row_count"),
        "column_count": v3_input.get("column_count"),
        "sha256": v3_input.get("physical_sha256"),
        "manifest": cfg.v3_manifest,
    }
    return result


def _archive_member_inventory(path: Path) -> list[dict[str, Any]]:
    with ZipFile(path) as archive:
        return [
            {
                "member": item.filename,
                "compressed_bytes": item.compress_size,
                "uncompressed_bytes": item.file_size,
                "crc32": f"{item.CRC:08x}",
            }
            for item in archive.infolist()
        ]


def _read_archive_metadata(path: Path) -> dict[str, str | None]:
    with ZipFile(path) as archive:
        try:
            metadata = archive.read("metadata.xml")
        except KeyError as exc:
            raise ValueError("DWCA has no metadata.xml") from exc
    root = ET.fromstring(metadata)
    title = _element_text(root, "title")
    if title is None:
        raise ValueError("DWCA metadata has no title")
    match = _DOWNLOAD_KEY.search(title)
    if match is None:
        raise ValueError("DWCA title has no GBIF download key")
    return {
        "download_key": match.group("key"),
        "package_id": root.attrib.get("packageId"),
        "title": title,
        "publication_date": _element_text(root, "pubDate"),
    }


def _element_text(root: ET.Element, local_name: str) -> str | None:
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == local_name and element.text:
            value = element.text.strip()
            if value:
                return value
    return None


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_sha256(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text.removeprefix("sha256:")


def _optional_int(value: object | None) -> int | None:
    return None if value is None else int(value)


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


__all__ = [
    "INVENTORY_SCHEMA",
    "INVENTORY_SCHEMA_VERSION",
    "REQUIRED_DWCA_MEMBERS",
    "SourceInventory",
    "SourceInventoryConfig",
    "build_source_inventory",
]
