from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

import pyarrow as pa

from biominer.common.semantic_hash import canonical_semantic_fingerprint


SCHEMA_VERSION = "biominer-gbif-media-url-resolution/v1"
JOB_NAME = "gbif_media_url_resolution"
STAGE = "resolve"


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    RIGHTS_BLOCKED = "rights_blocked"
    NON_IMAGE_MEDIA = "non_image_media"
    UNRESOLVED_NOT_FOUND = "unresolved_not_found"
    UNRESOLVED_ACCESS_DENIED = "unresolved_access_denied"
    UNRESOLVED_AMBIGUOUS_CANDIDATES = "unresolved_ambiguous_candidates"
    UNRESOLVED_INVALID_IMAGE = "unresolved_invalid_image"
    UNRESOLVED_PROVIDER_UNAVAILABLE = "unresolved_provider_unavailable"
    UNRESOLVED_ARCHIVE_REFERENCE_ONLY = "unresolved_archive_reference_only"
    RETRY_EXHAUSTED = "retry_exhausted"


def source_row_id(
    source_artifact_sha256: str,
    gbif_id: object,
    media_references: object,
) -> str:
    """Return the source-bound semantic identity for an affected media row."""

    return canonical_semantic_fingerprint(
        {
            "contract": "gbif-media-reference-row/v1",
            "source_artifact_sha256": str(source_artifact_sha256).strip(),
            "gbifID": str(gbif_id).strip(),
            "media_references": str(media_references).strip(),
        }
    )


def is_explicitly_restricted(value: object | None) -> bool:
    if value is None:
        return False
    normalized = str(value).strip().casefold()
    return "all rights reserved" in normalized or normalized == "copyright"


def license_basis(media_license: object | None, occurrence_license: object | None) -> str:
    if media_license is not None and str(media_license).strip():
        return "item_media_license"
    if occurrence_license is not None and str(occurrence_license).strip():
        return "occurrence_license_fallback"
    return "unknown"


@dataclass(frozen=True, slots=True)
class ResolutionInput:
    source_row_id: str
    source_artifact_sha256: str
    gbif_id: str
    media_references: str
    media_type: str | None
    media_format: str | None
    media_license: str | None
    occurrence_license: str | None
    provider: str | None = None
    publisher: str | None = None
    dataset_name: str | None = None
    taxon_rank: str | None = None
    country_code: str | None = None

    @property
    def host(self) -> str:
        return (urlsplit(self.media_references).hostname or "").casefold()

    @property
    def license_basis(self) -> str:
        return license_basis(self.media_license, self.occurrence_license)

    @property
    def rights_blocked(self) -> bool:
        return is_explicitly_restricted(self.media_license)

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_row_id": self.source_row_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "gbif_id": self.gbif_id,
            "media_references": self.media_references,
            "media_type": self.media_type,
            "media_format": self.media_format,
            "media_license": self.media_license,
            "occurrence_license": self.occurrence_license,
            "provider": self.provider,
            "publisher": self.publisher,
            "dataset_name": self.dataset_name,
            "taxon_rank": self.taxon_rank,
            "country_code": self.country_code,
        }

    @classmethod
    def from_payload(cls, value: dict[str, Any]) -> ResolutionInput:
        return cls(
            source_row_id=str(value["source_row_id"]),
            source_artifact_sha256=str(value["source_artifact_sha256"]),
            gbif_id=str(value["gbif_id"]),
            media_references=str(value["media_references"]),
            media_type=_optional_string(value.get("media_type")),
            media_format=_optional_string(value.get("media_format")),
            media_license=_optional_string(value.get("media_license")),
            occurrence_license=_optional_string(value.get("occurrence_license")),
            provider=_optional_string(value.get("provider")),
            publisher=_optional_string(value.get("publisher")),
            dataset_name=_optional_string(value.get("dataset_name")),
            taxon_rank=_optional_string(value.get("taxon_rank")),
            country_code=_optional_string(value.get("country_code")),
        )


@dataclass(frozen=True, slots=True)
class ResolutionAttempt:
    attempt_id: str
    source_row_id: str
    sequence: int
    phase: str
    method: str
    requested_url: str
    response_url: str | None
    redirect_from: str | None
    status_code: int | None
    outcome: str
    error: str | None
    declared_content_type: str | None
    response_prefix_sha256: str | None
    response_byte_count: int
    etag: str | None
    last_modified: str | None
    retry_number: int
    started_at: str
    ended_at: str

    def to_row(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    source_row_id: str
    source_artifact_sha256: str
    gbif_id: str
    media_references: str
    reference_host: str
    media_type: str | None
    media_format: str | None
    media_license: str | None
    occurrence_license: str | None
    license_basis: str
    status: ResolutionStatus
    method: str
    stable_candidate_url: str | None
    validated_final_url: str | None
    redirect_count: int
    declared_content_type: str | None
    detected_content_type: str | None
    bytes_sampled: int
    probe_prefix_sha256: str | None
    content_sha256: str | None
    content_hash_status: str
    adapter_version: str
    attempt_count: int
    terminal_reason: str | None
    resolved_at: str
    provenance_fingerprint: str

    def to_row(self) -> dict[str, Any]:
        row = {name: getattr(self, name) for name in self.__dataclass_fields__}
        row["status"] = self.status.value
        return row


ATTEMPT_SCHEMA = pa.schema(
    [
        ("attempt_id", pa.string()),
        ("source_row_id", pa.string()),
        ("sequence", pa.int32()),
        ("phase", pa.string()),
        ("method", pa.string()),
        ("requested_url", pa.string()),
        ("response_url", pa.string()),
        ("redirect_from", pa.string()),
        ("status_code", pa.int32()),
        ("outcome", pa.string()),
        ("error", pa.string()),
        ("declared_content_type", pa.string()),
        ("response_prefix_sha256", pa.string()),
        ("response_byte_count", pa.int64()),
        ("etag", pa.string()),
        ("last_modified", pa.string()),
        ("retry_number", pa.int32()),
        ("started_at", pa.string()),
        ("ended_at", pa.string()),
    ]
)

RESULT_SCHEMA = pa.schema(
    [
        ("source_row_id", pa.string()),
        ("source_artifact_sha256", pa.string()),
        ("gbif_id", pa.string()),
        ("media_references", pa.string()),
        ("reference_host", pa.string()),
        ("media_type", pa.string()),
        ("media_format", pa.string()),
        ("media_license", pa.string()),
        ("occurrence_license", pa.string()),
        ("license_basis", pa.string()),
        ("status", pa.string()),
        ("method", pa.string()),
        ("stable_candidate_url", pa.string()),
        ("validated_final_url", pa.string()),
        ("redirect_count", pa.int32()),
        ("declared_content_type", pa.string()),
        ("detected_content_type", pa.string()),
        ("bytes_sampled", pa.int64()),
        ("probe_prefix_sha256", pa.string()),
        ("content_sha256", pa.string()),
        ("content_hash_status", pa.string()),
        ("adapter_version", pa.string()),
        ("attempt_count", pa.int32()),
        ("terminal_reason", pa.string()),
        ("resolved_at", pa.string()),
        ("provenance_fingerprint", pa.string()),
    ]
)


def _optional_string(value: object | None) -> str | None:
    if value is None:
        return None
    result = str(value)
    return result if result else None
