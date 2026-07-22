from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq


PROVIDER_ENRICHMENT_VERSION = "biominer-gbif-provider-enrichment/v1"


@dataclass(frozen=True, slots=True)
class ProviderMetadataRequest:
    adapter_id: str
    adapter_version: str
    provider: str
    request_mode: str
    resource_identifier: str | None
    source_uri: str | None
    cache_key: str


@dataclass(frozen=True, slots=True)
class ProviderMetadataEvidence:
    adapter_id: str
    adapter_version: str
    evidence_source_uri: str
    evidence_scope: str
    provider_resource_identifier: str | None
    media_license: str | None
    creator: str | None
    rights_holder: str | None
    effective_from: str | None
    effective_to: str | None


@runtime_checkable
class ProviderMetadataAdapter(Protocol):
    adapter_id: str
    version: str
    priority: int

    def supports(self, provider: str) -> bool: ...

    def request(
        self, *, provider: str, resource_identifier: str | None
    ) -> ProviderMetadataRequest: ...

    def parse(
        self, record: Mapping[str, object], *, evidence_source_uri: str
    ) -> ProviderMetadataEvidence: ...


class StructuredProviderMetadataAdapter:
    """Provider contract that accepts only direct, structured rights evidence."""

    def __init__(
        self,
        *,
        adapter_id: str,
        version: str,
        priority: int,
        provider_name: str,
        aliases: tuple[str, ...] = (),
        source_uri: str | None = None,
        request_mode: str = "bulk_export_preferred",
    ) -> None:
        self.adapter_id = adapter_id
        self.version = version
        self.priority = priority
        self.provider_name = provider_name
        self.aliases = aliases
        self.source_uri = source_uri
        self.request_mode = request_mode
        self._names = {value.casefold().strip() for value in (provider_name, *aliases)}

    def supports(self, provider: str) -> bool:
        return provider.casefold().strip() in self._names

    def request(
        self, *, provider: str, resource_identifier: str | None
    ) -> ProviderMetadataRequest:
        if not self.supports(provider):
            raise ValueError(f"adapter {self.adapter_id} does not support {provider}")
        identity = "|".join(
            (self.adapter_id, self.version, provider.strip(), resource_identifier or "")
        )
        return ProviderMetadataRequest(
            adapter_id=self.adapter_id,
            adapter_version=self.version,
            provider=provider.strip(),
            request_mode=self.request_mode,
            resource_identifier=_text(resource_identifier),
            source_uri=self.source_uri,
            cache_key="sha256:" + hashlib.sha256(identity.encode()).hexdigest(),
        )

    def parse(
        self, record: Mapping[str, object], *, evidence_source_uri: str
    ) -> ProviderMetadataEvidence:
        if not evidence_source_uri.startswith(("https://", "http://", "file://")):
            raise ValueError("provider evidence requires an explicit source URI")
        scope = _text(record.get("evidence_scope"))
        if scope not in {"item", "collection"}:
            raise ValueError("provider evidence scope must be item or collection")
        return ProviderMetadataEvidence(
            adapter_id=self.adapter_id,
            adapter_version=self.version,
            evidence_source_uri=evidence_source_uri,
            evidence_scope=scope,
            provider_resource_identifier=_text(record.get("provider_resource_identifier")),
            media_license=_text(record.get("media_license")),
            creator=_text(record.get("creator")),
            rights_holder=_text(record.get("rights_holder")),
            effective_from=_text(record.get("effective_from")),
            effective_to=_text(record.get("effective_to")),
        )


DEFAULT_PROVIDER_METADATA_ADAPTERS: tuple[StructuredProviderMetadataAdapter, ...] = tuple(
    StructuredProviderMetadataAdapter(
        adapter_id=adapter_id,
        version=f"{adapter_id}/v1",
        priority=priority,
        provider_name=provider,
    )
    for priority, (adapter_id, provider) in enumerate(
        (
            ("vermont_center_for_ecostudies", "Vermont Center for Ecostudies"),
            ("tiroler_landesmuseum", "Tiroler Landesmuseum Ferdinandeum"),
            ("fotografia_y_biodiversidad", "Fotografía y Biodiversidad"),
            ("danish_sgav", "Danish SGAV"),
            ("international_barcode_of_life", "International Barcode of Life"),
            ("swissnatcoll", "SwissNatColl"),
            ("naturemapr", "NatureMapr"),
        ),
        start=1,
    )
)


REGISTRY_SCHEMA = pa.schema(
    [
        ("provider_enrichment_version", pa.string()),
        ("priority", pa.int32()),
        ("adapter_id", pa.string()),
        ("adapter_version", pa.string()),
        ("provider_name", pa.string()),
        ("request_mode", pa.string()),
        ("source_uri", pa.string()),
        ("execution_status", pa.string()),
        ("evidence_scope_status", pa.string()),
        ("network_requests", pa.int64()),
    ]
)


def publish_provider_enrichment_registry(
    *, output_directory: str | Path, source_snapshot_id: str, code_commit: str
) -> dict[str, object]:
    """Publish the seven prioritized adapters without claiming enrichment execution."""

    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    output = staging / "provider_enrichment_registry.parquet"
    rows = [
        {
            "provider_enrichment_version": PROVIDER_ENRICHMENT_VERSION,
            "priority": adapter.priority,
            "adapter_id": adapter.adapter_id,
            "adapter_version": adapter.version,
            "provider_name": adapter.provider_name,
            "request_mode": adapter.request_mode,
            "source_uri": adapter.source_uri,
            "execution_status": "NOT_TESTED",
            "evidence_scope_status": "UNKNOWN",
            "network_requests": 0,
        }
        for adapter in DEFAULT_PROVIDER_METADATA_ADAPTERS
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=REGISTRY_SCHEMA), output)
    artifact = _artifact(output)
    manifest = {
        "schema_version": PROVIDER_ENRICHMENT_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "code_commit": code_commit,
        "source_snapshot_id": source_snapshot_id,
        "counts": {"registered_adapters": len(rows), "executed_adapters": 0},
        "validation": {
            "seven_prioritized_providers_registered": len(rows) == 7,
            "adapter_versions_present": all(row["adapter_version"] for row in rows),
            "network_claims_withheld": True,
            "manifest_written_last": True,
        },
        "artifacts": [artifact],
        "network_requests": 0,
        "manifest_policy": {"written_last": True},
    }
    _write_json(staging / "manifest.json", manifest)
    if _sha256(output) != artifact["sha256"]:
        raise ValueError("provider registry checksum mismatch")
    os.replace(staging, destination)
    return manifest


def _text(value: object | None) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _artifact(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    return {
        "path": path.name,
        "physical_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "row_count": parquet.metadata.num_rows,
        "column_count": len(parquet.schema_arrow),
        "row_group_count": parquet.metadata.num_row_groups,
    }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DEFAULT_PROVIDER_METADATA_ADAPTERS",
    "PROVIDER_ENRICHMENT_VERSION",
    "ProviderMetadataAdapter",
    "ProviderMetadataEvidence",
    "ProviderMetadataRequest",
    "StructuredProviderMetadataAdapter",
    "publish_provider_enrichment_registry",
]
