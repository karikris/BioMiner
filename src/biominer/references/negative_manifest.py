from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import polars as pl

from biominer.references.licensing import ReferenceLicencePolicy
from biominer.reports.flickr_fetch import current_git_sha
from biominer.storage.parquet import write_parquet


CURATED_VISUAL_DOMAIN_NEGATIVE_SOURCE_SCHEMA_VERSION = (
    "curated-visual-domain-negative-source-v1.0.0"
)
REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_SCHEMA_VERSION = (
    "reference-visual-domain-negative-manifest-v1.0.0"
)
REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_REPORT_SCHEMA_VERSION = (
    "reference-visual-domain-negative-manifest-report-v1.0.0"
)

REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_FILE = (
    "reference_visual_domain_negative_manifest.parquet"
)
REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_REPORT_FILE = (
    "reference_visual_domain_negative_manifest_report.json"
)

NEGATIVE_CATEGORIES = frozenset(
    {
        "artwork",
        "logo",
        "tattoo",
        "non_butterfly_insect_illustration",
        "partial_wing",
        "misleading_pattern",
    }
)
NEGATIVE_SOURCE_KINDS = frozenset(
    {
        "institutional_repository",
        "licensed_media_repository",
        "creator_supplied",
        "commissioned",
        "internal_original",
    }
)
NEGATIVE_TARGET_PRESENCE_VALUES = frozenset({"present", "absent", "unknown"})
NEGATIVE_REVIEW_STATUSES = frozenset({"pending", "verified", "excluded"})
NEGATIVE_REVIEW_CONFIDENCE_VALUES = frozenset({"high", "medium", "low", "unknown"})

_VISUAL_DOMAIN_BY_CATEGORY = {
    "artwork": "artwork",
    "logo": "logo",
    "tattoo": "tattoo",
    "non_butterfly_insect_illustration": "artwork",
    "partial_wing": "partial_wing",
    "misleading_pattern": "unsuitable",
}
_SOURCE_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_version",
        "source_snapshot_version",
        "target_accepted_taxon_key",
        "negatives",
    }
)
_SOURCE_ROW_FIELDS = frozenset(
    {
        "source_kind",
        "source",
        "source_record_id",
        "provider_media_id",
        "source_record_uri",
        "media_uri",
        "source_sha256",
        "negative_category",
        "target_presence",
        "creator",
        "creator_uri",
        "rights_holder",
        "licence",
        "licence_uri",
        "attribution",
        "rights_evidence_uri",
        "review_status",
        "reviewed_by",
        "reviewed_at",
        "review_confidence",
        "review_notes",
        "exclusion_reason",
        "enabled",
    }
)
_NULLABLE_SOURCE_ROW_FIELDS = frozenset(
    {
        "source_sha256",
        "creator_uri",
        "reviewed_by",
        "reviewed_at",
        "review_notes",
        "exclusion_reason",
    }
)
_SORT_FIELDS = [
    "negative_category",
    "source_kind",
    "source",
    "source_record_id",
    "provider_media_id",
    "negative_reference_id",
]
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NEGATIVE_REFERENCE_ID_PATTERN = re.compile(r"reference-negative:[0-9a-f]{64}\Z")
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_LOGGER = logging.getLogger(__name__)


def curated_visual_domain_negative_manifest_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "negative_reference_id": pl.String,
        "manifest_version": pl.String,
        "source_snapshot_version": pl.String,
        "target_accepted_taxon_key": pl.String,
        "source_kind": pl.String,
        "source": pl.String,
        "source_record_id": pl.String,
        "provider_media_id": pl.String,
        "source_record_uri": pl.String,
        "media_uri": pl.String,
        "source_sha256": pl.String,
        "negative_category": pl.String,
        "visual_domain": pl.String,
        "target_presence": pl.String,
        "creator": pl.String,
        "creator_uri": pl.String,
        "rights_holder": pl.String,
        "licence": pl.String,
        "licence_uri": pl.String,
        "attribution": pl.String,
        "rights_evidence_uri": pl.String,
        "canonical_licence": pl.String,
        "licence_policy_status": pl.String,
        "licence_policy_reason": pl.String,
        "licence_policy_version": pl.String,
        "licence_policy_fingerprint": pl.String,
        "review_status": pl.String,
        "reviewed_by": pl.String,
        "reviewed_at": pl.Datetime("us", "UTC"),
        "review_confidence": pl.String,
        "review_notes": pl.String,
        "exclusion_reason": pl.String,
        "enabled": pl.Boolean,
        "row_fingerprint": pl.String,
    }


def load_curated_visual_domain_negative_source(
    source: str | Path | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(source, Mapping):
        payload: object = deepcopy(dict(source))
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size > _MAX_SOURCE_BYTES:
            raise ValueError(
                f"curated negative source exceeds {_MAX_SOURCE_BYTES} bytes: {path}"
            )
        try:
            payload = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except UnicodeDecodeError as exc:
            raise ValueError(f"curated negative source is not UTF-8: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid curated negative JSON: {path}") from exc
    else:
        raise TypeError("source must be a local path or mapping")

    if not isinstance(payload, dict):
        raise TypeError("curated negative source root must be an object")
    _require_exact_fields(payload, _SOURCE_ROOT_FIELDS, field="source root")
    if (
        payload["schema_version"]
        != CURATED_VISUAL_DOMAIN_NEGATIVE_SOURCE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported curated negative source schema: {payload['schema_version']!r}"
        )
    _required_text(payload["manifest_version"], field="manifest_version")
    _required_text(payload["source_snapshot_version"], field="source_snapshot_version")
    _target_taxon_key(payload["target_accepted_taxon_key"])
    negatives = payload["negatives"]
    if not isinstance(negatives, list):
        raise TypeError("curated negative source negatives must be a list")
    if not negatives:
        raise ValueError("curated negative source must contain at least one negative")
    return payload


def compile_curated_visual_domain_negative_manifest(
    source: str | Path | Mapping[str, object],
    *,
    licence_policy: ReferenceLicencePolicy | None = None,
) -> pl.DataFrame:
    policy = licence_policy or ReferenceLicencePolicy()
    if not isinstance(policy, ReferenceLicencePolicy):
        raise TypeError("licence_policy must be a ReferenceLicencePolicy")
    payload = load_curated_visual_domain_negative_source(source)
    root = {
        "manifest_version": _required_text(
            payload["manifest_version"], field="manifest_version"
        ),
        "source_snapshot_version": _required_text(
            payload["source_snapshot_version"], field="source_snapshot_version"
        ),
        "target_accepted_taxon_key": _target_taxon_key(
            payload["target_accepted_taxon_key"]
        ),
    }
    rows: list[dict[str, object]] = []
    negatives = payload["negatives"]
    assert isinstance(negatives, list)
    for index, value in enumerate(negatives):
        if not isinstance(value, Mapping):
            raise TypeError(f"negatives[{index}] must be an object")
        rows.append(
            _compile_negative_row(
                value,
                index=index,
                root=root,
                licence_policy=policy,
            )
        )
    schema = curated_visual_domain_negative_manifest_schema()
    frame = (
        pl.DataFrame(rows, schema=schema, strict=True).sort(_SORT_FIELDS)
        if rows
        else pl.DataFrame(schema=schema)
    )
    validate_curated_visual_domain_negative_manifest(
        frame,
        licence_policy=policy,
    )
    return frame


def validate_curated_visual_domain_negative_manifest(
    frame: pl.DataFrame,
    *,
    licence_policy: ReferenceLicencePolicy | None = None,
) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a Polars DataFrame")
    policy = licence_policy or ReferenceLicencePolicy()
    if not isinstance(policy, ReferenceLicencePolicy):
        raise TypeError("licence_policy must be a ReferenceLicencePolicy")
    schema = curated_visual_domain_negative_manifest_schema()
    if frame.schema != schema:
        raise ValueError("curated negative manifest does not match the physical schema")
    if not frame.equals(frame.sort(_SORT_FIELDS)):
        raise ValueError("curated negative manifest is not deterministically sorted")

    _reject_duplicates(frame, ["negative_reference_id"], "negative reference IDs")
    _reject_duplicates(
        frame,
        ["source", "source_record_id", "provider_media_id"],
        "source media identities",
    )
    _reject_duplicates(frame, ["media_uri"], "media URIs")
    nonnull_checksums = frame.filter(pl.col("source_sha256").is_not_null())
    _reject_duplicates(nonnull_checksums, ["source_sha256"], "source SHA-256 values")

    root_identity: tuple[str, str, str] | None = None
    for index, row in enumerate(frame.iter_rows(named=True)):
        _validate_compiled_row(row, index=index, licence_policy=policy)
        identity = (
            str(row["manifest_version"]),
            str(row["source_snapshot_version"]),
            str(row["target_accepted_taxon_key"]),
        )
        if root_identity is None:
            root_identity = identity
        elif identity != root_identity:
            raise ValueError("curated negative manifest mixes root manifest identities")


def write_curated_visual_domain_negative_manifest(
    frame: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = True,
    licence_policy: ReferenceLicencePolicy | None = None,
) -> Path:
    validate_curated_visual_domain_negative_manifest(
        frame,
        licence_policy=licence_policy,
    )
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_FILE
    return write_parquet(frame, destination, overwrite=overwrite)


def publish_curated_visual_domain_negative_manifest(
    source: str | Path | Mapping[str, object],
    output_dir: str | Path,
    *,
    licence_policy: ReferenceLicencePolicy | None = None,
    run_id: str | None = None,
) -> dict[str, Path]:
    started_at = datetime.now(UTC)
    effective_run_id = _required_text(
        run_id
        or (
            "reference-negative-manifest-"
            + started_at.strftime("%Y%m%dT%H%M%S%fZ-")
            + uuid4().hex[:12]
        ),
        field="run_id",
    )
    directory = Path(output_dir).resolve()
    staging = directory.parent / f".{directory.name}.{uuid4().hex}.tmp"
    _log_event(
        "reference_visual_domain_negative_publication_started",
        command="references.compile_visual_domain_negatives",
        run_id=effective_run_id,
        output_dir=str(directory),
        row_count=None,
        started_at=started_at.isoformat(),
    )
    lock_fd: int | None = None
    try:
        policy = licence_policy or ReferenceLicencePolicy()
        payload = load_curated_visual_domain_negative_source(source)
        frame = compile_curated_visual_domain_negative_manifest(
            payload,
            licence_policy=policy,
        )
        directory.parent.mkdir(parents=True, exist_ok=True)
        lock_path = directory.parent / f".{directory.name}.publish.lock"
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FileExistsError(directory) from exc
        if directory.exists():
            raise FileExistsError(directory)
        staging.mkdir(parents=False, exist_ok=False)
        staged_manifest = write_curated_visual_domain_negative_manifest(
            frame,
            staging / REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_FILE,
            overwrite=False,
            licence_policy=policy,
        )
        final_manifest = directory / REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_FILE
        artifact = {
            "uri": str(final_manifest),
            "row_count": frame.height,
            "byte_count": staged_manifest.stat().st_size,
            "sha256": _file_sha256(staged_manifest),
        }
        ended_at = datetime.now(UTC)
        report = _publication_report(
            frame,
            policy=policy,
            artifact=artifact,
            run_id=effective_run_id,
            started_at=started_at,
            ended_at=ended_at,
            source_fingerprint=_source_fingerprint(payload),
        )
        staged_report = staging / REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_REPORT_FILE
        staged_report.write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        staging.replace(directory)
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        ended_at = datetime.now(UTC)
        _log_event(
            "reference_visual_domain_negative_publication_failed",
            command="references.compile_visual_domain_negatives",
            run_id=effective_run_id,
            output_dir=str(directory),
            ended_at=ended_at.isoformat(),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        _write_failed_publication_audit(
            directory,
            run_id=effective_run_id,
            started_at=started_at,
            ended_at=ended_at,
            error=exc,
        )
        raise
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
    _log_event(
        "reference_visual_domain_negative_publication_completed",
        command="references.compile_visual_domain_negatives",
        run_id=effective_run_id,
        output_dir=str(directory),
        row_count=frame.height,
        ended_at=ended_at.isoformat(),
    )
    return {
        "manifest": directory / REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_FILE,
        "report": directory / REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_REPORT_FILE,
    }


@dataclass(frozen=True, slots=True)
class CuratedVisualDomainNegativeSourceAdapter:
    licence_policy: ReferenceLicencePolicy = field(
        default_factory=ReferenceLicencePolicy
    )

    def __post_init__(self) -> None:
        if not isinstance(self.licence_policy, ReferenceLicencePolicy):
            raise TypeError("licence_policy must be a ReferenceLicencePolicy")

    def load(
        self,
        source: str | Path | Mapping[str, object],
    ) -> dict[str, object]:
        return load_curated_visual_domain_negative_source(source)

    def compile(
        self,
        source: str | Path | Mapping[str, object],
    ) -> pl.DataFrame:
        return compile_curated_visual_domain_negative_manifest(
            source,
            licence_policy=self.licence_policy,
        )

    def validate(self, frame: pl.DataFrame) -> None:
        validate_curated_visual_domain_negative_manifest(
            frame,
            licence_policy=self.licence_policy,
        )

    def write(
        self,
        frame: pl.DataFrame,
        output: str | Path,
        *,
        overwrite: bool = True,
    ) -> Path:
        return write_curated_visual_domain_negative_manifest(
            frame,
            output,
            overwrite=overwrite,
            licence_policy=self.licence_policy,
        )

    def publish(
        self,
        source: str | Path | Mapping[str, object],
        output_dir: str | Path,
        *,
        run_id: str | None = None,
    ) -> dict[str, Path]:
        return publish_curated_visual_domain_negative_manifest(
            source,
            output_dir,
            licence_policy=self.licence_policy,
            run_id=run_id,
        )


def _compile_negative_row(
    value: Mapping[str, object],
    *,
    index: int,
    root: Mapping[str, str],
    licence_policy: ReferenceLicencePolicy,
) -> dict[str, object]:
    field_prefix = f"negatives[{index}]"
    _require_exact_fields(value, _SOURCE_ROW_FIELDS, field=field_prefix)
    for name in sorted(_SOURCE_ROW_FIELDS - _NULLABLE_SOURCE_ROW_FIELDS):
        if value[name] is None:
            raise TypeError(f"{field_prefix}.{name} cannot be null")

    source_kind = _choice(
        value["source_kind"],
        field=f"{field_prefix}.source_kind",
        choices=NEGATIVE_SOURCE_KINDS,
    )
    source_name = _required_text(value["source"], field=f"{field_prefix}.source")
    source_record_id = _required_text(
        value["source_record_id"], field=f"{field_prefix}.source_record_id"
    )
    provider_media_id = _required_text(
        value["provider_media_id"], field=f"{field_prefix}.provider_media_id"
    )
    source_record_uri = _absolute_uri(
        value["source_record_uri"], field=f"{field_prefix}.source_record_uri"
    )
    media_uri = _absolute_uri(value["media_uri"], field=f"{field_prefix}.media_uri")
    source_sha256 = _nullable_sha256(
        value["source_sha256"], field=f"{field_prefix}.source_sha256"
    )
    negative_category = _choice(
        value["negative_category"],
        field=f"{field_prefix}.negative_category",
        choices=NEGATIVE_CATEGORIES,
    )
    target_presence = _choice(
        value["target_presence"],
        field=f"{field_prefix}.target_presence",
        choices=NEGATIVE_TARGET_PRESENCE_VALUES,
    )
    creator = _required_text(value["creator"], field=f"{field_prefix}.creator")
    creator_uri = _nullable_uri(
        value["creator_uri"], field=f"{field_prefix}.creator_uri"
    )
    rights_holder = _required_text(
        value["rights_holder"], field=f"{field_prefix}.rights_holder"
    )
    licence = _required_text(value["licence"], field=f"{field_prefix}.licence")
    licence_uri = _absolute_uri(
        value["licence_uri"], field=f"{field_prefix}.licence_uri"
    )
    attribution = _required_text(
        value["attribution"], field=f"{field_prefix}.attribution"
    )
    rights_evidence_uri = _absolute_uri(
        value["rights_evidence_uri"],
        field=f"{field_prefix}.rights_evidence_uri",
    )
    review_status = _choice(
        value["review_status"],
        field=f"{field_prefix}.review_status",
        choices=NEGATIVE_REVIEW_STATUSES,
    )
    reviewed_by = _nullable_text(
        value["reviewed_by"], field=f"{field_prefix}.reviewed_by"
    )
    reviewed_at = _nullable_utc_datetime(
        value["reviewed_at"], field=f"{field_prefix}.reviewed_at"
    )
    review_confidence = _choice(
        value["review_confidence"],
        field=f"{field_prefix}.review_confidence",
        choices=NEGATIVE_REVIEW_CONFIDENCE_VALUES,
    )
    review_notes = _nullable_text(
        value["review_notes"], field=f"{field_prefix}.review_notes"
    )
    exclusion_reason = _nullable_text(
        value["exclusion_reason"], field=f"{field_prefix}.exclusion_reason"
    )
    enabled = value["enabled"]
    if not isinstance(enabled, bool):
        raise TypeError(f"{field_prefix}.enabled must be Boolean")
    _validate_review_state(
        review_status=review_status,
        reviewed_by=reviewed_by,
        reviewed_at=reviewed_at,
        review_confidence=review_confidence,
        exclusion_reason=exclusion_reason,
        enabled=enabled,
        field=field_prefix,
    )

    decision = licence_policy.evaluate(
        media_licence=licence,
        licence_uri=licence_uri,
        attribution=attribution,
    )
    if enabled and decision.status != "allowed":
        raise ValueError(
            f"{field_prefix}.enabled requires an allowed licence policy decision"
        )
    identity = {
        "source_kind": source_kind,
        "source": source_name,
        "source_record_id": source_record_id,
        "provider_media_id": provider_media_id,
    }
    row: dict[str, object] = {
        "schema_version": REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_SCHEMA_VERSION,
        **root,
        "negative_reference_id": "reference-negative:" + _canonical_sha256(identity),
        "source_kind": source_kind,
        "source": source_name,
        "source_record_id": source_record_id,
        "provider_media_id": provider_media_id,
        "source_record_uri": source_record_uri,
        "media_uri": media_uri,
        "source_sha256": source_sha256,
        "negative_category": negative_category,
        "visual_domain": _VISUAL_DOMAIN_BY_CATEGORY[negative_category],
        "target_presence": target_presence,
        "creator": creator,
        "creator_uri": creator_uri,
        "rights_holder": rights_holder,
        "licence": licence,
        "licence_uri": licence_uri,
        "canonical_licence": decision.canonical_licence,
        "attribution": attribution,
        "rights_evidence_uri": rights_evidence_uri,
        "review_status": review_status,
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
        "review_confidence": review_confidence,
        "review_notes": review_notes,
        "exclusion_reason": exclusion_reason,
        "enabled": enabled,
        "licence_policy_status": decision.status,
        "licence_policy_reason": decision.reason,
        "licence_policy_version": licence_policy.version,
        "licence_policy_fingerprint": licence_policy.fingerprint,
    }
    row["row_fingerprint"] = "sha256:" + _canonical_sha256(row)
    return row


def _validate_compiled_row(
    row: Mapping[str, object],
    *,
    index: int,
    licence_policy: ReferenceLicencePolicy,
) -> None:
    field_prefix = f"manifest row {index}"
    if (
        row["schema_version"]
        != REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(f"{field_prefix} has an incompatible schema version")
    _required_text(row["manifest_version"], field=f"{field_prefix}.manifest_version")
    _required_text(
        row["source_snapshot_version"],
        field=f"{field_prefix}.source_snapshot_version",
    )
    _target_taxon_key(row["target_accepted_taxon_key"])
    source_kind = _choice(
        row["source_kind"],
        field=f"{field_prefix}.source_kind",
        choices=NEGATIVE_SOURCE_KINDS,
    )
    source_name = _required_text(row["source"], field=f"{field_prefix}.source")
    source_record_id = _required_text(
        row["source_record_id"], field=f"{field_prefix}.source_record_id"
    )
    provider_media_id = _required_text(
        row["provider_media_id"], field=f"{field_prefix}.provider_media_id"
    )
    _absolute_uri(row["source_record_uri"], field=f"{field_prefix}.source_record_uri")
    _absolute_uri(row["media_uri"], field=f"{field_prefix}.media_uri")
    _nullable_sha256(row["source_sha256"], field=f"{field_prefix}.source_sha256")
    negative_category = _choice(
        row["negative_category"],
        field=f"{field_prefix}.negative_category",
        choices=NEGATIVE_CATEGORIES,
    )
    if row["visual_domain"] != _VISUAL_DOMAIN_BY_CATEGORY[negative_category]:
        raise ValueError(f"{field_prefix}.visual_domain is inconsistent with category")
    _choice(
        row["target_presence"],
        field=f"{field_prefix}.target_presence",
        choices=NEGATIVE_TARGET_PRESENCE_VALUES,
    )
    for name in ("creator", "rights_holder", "licence", "attribution"):
        _required_text(row[name], field=f"{field_prefix}.{name}")
    _nullable_uri(row["creator_uri"], field=f"{field_prefix}.creator_uri")
    _absolute_uri(row["licence_uri"], field=f"{field_prefix}.licence_uri")
    _absolute_uri(
        row["rights_evidence_uri"], field=f"{field_prefix}.rights_evidence_uri"
    )
    review_status = _choice(
        row["review_status"],
        field=f"{field_prefix}.review_status",
        choices=NEGATIVE_REVIEW_STATUSES,
    )
    reviewed_by = _nullable_text(
        row["reviewed_by"], field=f"{field_prefix}.reviewed_by"
    )
    reviewed_at = _nullable_utc_datetime(
        row["reviewed_at"], field=f"{field_prefix}.reviewed_at"
    )
    review_confidence = _choice(
        row["review_confidence"],
        field=f"{field_prefix}.review_confidence",
        choices=NEGATIVE_REVIEW_CONFIDENCE_VALUES,
    )
    _nullable_text(row["review_notes"], field=f"{field_prefix}.review_notes")
    exclusion_reason = _nullable_text(
        row["exclusion_reason"], field=f"{field_prefix}.exclusion_reason"
    )
    enabled = row["enabled"]
    if not isinstance(enabled, bool):
        raise TypeError(f"{field_prefix}.enabled must be Boolean")
    _validate_review_state(
        review_status=review_status,
        reviewed_by=reviewed_by,
        reviewed_at=reviewed_at,
        review_confidence=review_confidence,
        exclusion_reason=exclusion_reason,
        enabled=enabled,
        field=field_prefix,
    )
    decision = licence_policy.evaluate(
        media_licence=row["licence"],
        licence_uri=row["licence_uri"],
        attribution=row["attribution"],
    )
    if (
        row["canonical_licence"] != decision.canonical_licence
        or row["licence_policy_status"] != decision.status
        or row["licence_policy_reason"] != decision.reason
        or row["licence_policy_version"] != licence_policy.version
        or row["licence_policy_fingerprint"] != licence_policy.fingerprint
    ):
        raise ValueError(f"{field_prefix} licence policy projection is stale")
    if enabled and decision.status != "allowed":
        raise ValueError(f"{field_prefix}.enabled requires an allowed licence")
    expected_id = "reference-negative:" + _canonical_sha256(
        {
            "source_kind": source_kind,
            "source": source_name,
            "source_record_id": source_record_id,
            "provider_media_id": provider_media_id,
        }
    )
    if (
        _NEGATIVE_REFERENCE_ID_PATTERN.fullmatch(str(row["negative_reference_id"]))
        is None
        or row["negative_reference_id"] != expected_id
    ):
        raise ValueError(f"{field_prefix} negative reference ID is invalid")
    row_without_fingerprint = {
        name: value for name, value in row.items() if name != "row_fingerprint"
    }
    expected_fingerprint = "sha256:" + _canonical_sha256(row_without_fingerprint)
    if (
        _SHA256_PATTERN.fullmatch(str(row["row_fingerprint"])) is None
        or row["row_fingerprint"] != expected_fingerprint
    ):
        raise ValueError(f"{field_prefix} row fingerprint is invalid")


def _publication_report(
    frame: pl.DataFrame,
    *,
    policy: ReferenceLicencePolicy,
    artifact: Mapping[str, object],
    run_id: str,
    started_at: datetime,
    ended_at: datetime,
    source_fingerprint: str,
) -> dict[str, Any]:
    def counts(field_name: str) -> dict[str, int]:
        return dict(
            sorted(
                Counter(str(value) for value in frame.get_column(field_name)).items()
            )
        )

    root = (
        frame.select(
            "manifest_version",
            "source_snapshot_version",
            "target_accepted_taxon_key",
        ).row(0, named=True)
        if frame.height
        else {
            "manifest_version": None,
            "source_snapshot_version": None,
            "target_accepted_taxon_key": None,
        }
    )
    return {
        "schema_version": REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_REPORT_SCHEMA_VERSION,
        "command": "references.compile_visual_domain_negatives",
        "run_id": run_id,
        "status": "complete",
        "git_sha": current_git_sha(),
        "pid": os.getpid(),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": max(0.0, (ended_at - started_at).total_seconds()),
        "network_requests": 0,
        "source_fingerprint": source_fingerprint,
        "row_count": frame.height,
        "enabled_count": frame.filter(pl.col("enabled")).height,
        "counts_by_category": counts("negative_category"),
        "counts_by_review_status": counts("review_status"),
        "counts_by_licence_policy_status": counts("licence_policy_status"),
        **root,
        "licence_policy_version": policy.version,
        "licence_policy_fingerprint": policy.fingerprint,
        "artifact": dict(artifact),
    }


def _write_failed_publication_audit(
    directory: Path,
    *,
    run_id: str,
    started_at: datetime,
    ended_at: datetime,
    error: Exception,
) -> None:
    audit_path = directory.parent / f".{directory.name}.{uuid4().hex}.failed.json"
    temporary_path = audit_path.with_suffix(".json.tmp")
    try:
        report = {
            "schema_version": (
                REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_REPORT_SCHEMA_VERSION
            ),
            "command": "references.compile_visual_domain_negatives",
            "run_id": run_id,
            "pid": os.getpid(),
            "git_sha": current_git_sha(),
            "status": "failed",
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "elapsed_seconds": max(0.0, (ended_at - started_at).total_seconds()),
            "network_requests": 0,
            "output_dir": str(directory),
            "error_type": type(error).__name__,
            "error": str(error),
            "artifact": "not_committed",
        }
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(audit_path)
    except Exception:  # noqa: BLE001 - audit failure must not mask the root failure.
        temporary_path.unlink(missing_ok=True)
        _LOGGER.exception(
            "could not persist failed curated negative publication audit",
            extra={"output_dir": str(directory), "run_id": run_id},
        )


def _validate_review_state(
    *,
    review_status: str,
    reviewed_by: str | None,
    reviewed_at: datetime | None,
    review_confidence: str,
    exclusion_reason: str | None,
    enabled: bool,
    field: str,
) -> None:
    if review_status == "pending":
        if review_confidence != "unknown":
            raise ValueError(f"{field}: pending review confidence must be unknown")
        if reviewed_by is not None or reviewed_at is not None:
            raise ValueError(f"{field}: pending review cannot have review provenance")
        if enabled:
            raise ValueError(f"{field}: pending review must be disabled")
        if exclusion_reason is not None:
            raise ValueError(f"{field}: pending review cannot have an exclusion reason")
    elif review_status == "verified":
        if review_confidence == "unknown":
            raise ValueError(f"{field}: verified review confidence cannot be unknown")
        if reviewed_by is None or reviewed_at is None:
            raise ValueError(f"{field}: verified review requires review provenance")
        if exclusion_reason is not None:
            raise ValueError(
                f"{field}: verified review cannot have an exclusion reason"
            )
    else:
        if review_confidence == "unknown":
            raise ValueError(f"{field}: excluded review confidence cannot be unknown")
        if reviewed_by is None or reviewed_at is None or exclusion_reason is None:
            raise ValueError(
                f"{field}: excluded review requires provenance and exclusion reason"
            )
        if enabled:
            raise ValueError(f"{field}: excluded review must be disabled")
    if enabled and review_confidence not in {"high", "medium"}:
        raise ValueError(f"{field}: enabled review confidence must be high or medium")


def _reject_duplicates(
    frame: pl.DataFrame,
    fields: Sequence[str],
    description: str,
) -> None:
    if frame.is_empty():
        return
    duplicates = frame.group_by(list(fields)).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ValueError(f"curated negative manifest contains duplicate {description}")


def _require_exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    field: str,
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"{field} is missing fields: {missing}")
    if unknown:
        raise ValueError(f"{field} has unknown fields: {unknown}")


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    if text != value:
        raise ValueError(f"{field} must not have surrounding whitespace")
    return text


def _nullable_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _choice(value: object, *, field: str, choices: frozenset[str]) -> str:
    text = _required_text(value, field=field)
    if text not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}")
    return text


def _target_taxon_key(value: object) -> str:
    text = _required_text(value, field="target_accepted_taxon_key")
    if not text.isascii() or not text.isdigit():
        raise ValueError("target_accepted_taxon_key must contain ASCII digits only")
    return text


def _absolute_uri(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if any(character.isspace() or ord(character) < 0x20 for character in text):
        raise ValueError(f"{field} contains whitespace or control characters")
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid absolute URI") from exc
    if (
        parsed.scheme not in {"http", "https", "s3"}
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            f"{field} must be an absolute http, https, or s3 URI without userinfo"
        )
    if parsed.scheme == "s3":
        if (
            port is not None
            or parsed.query
            or parsed.fragment
            or parsed.path in {"", "/"}
        ):
            raise ValueError(f"{field} must identify an s3 object")
    elif port is not None and not 1 <= port <= 65535:
        raise ValueError(f"{field} contains an invalid port")
    return text


def _nullable_uri(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _absolute_uri(value, field=field)


def _nullable_sha256(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 digest")
    return text


def _nullable_utc_datetime(value: object, *, field: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = _required_text(value, field=field)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO 8601 timestamp") from exc
    else:
        raise TypeError(f"{field} must be an ISO 8601 string, datetime, or null")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _canonical_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_fingerprint(payload: Mapping[str, object]) -> str:
    negatives = payload["negatives"]
    assert isinstance(negatives, list)
    canonical_rows = sorted(
        (dict(row) for row in negatives if isinstance(row, Mapping)),
        key=lambda row: json.dumps(
            row,
            default=_json_default,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
    )
    canonical_source = {
        "schema_version": payload["schema_version"],
        "manifest_version": payload["manifest_version"],
        "source_snapshot_version": payload["source_snapshot_version"],
        "target_accepted_taxon_key": payload["target_accepted_taxon_key"],
        "negatives": canonical_rows,
    }
    return "sha256:" + _canonical_sha256(canonical_source)


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise TypeError("cannot fingerprint a naive datetime")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    raise TypeError(f"unsupported fingerprint value: {type(value).__name__}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _log_event(event: str, **fields: object) -> None:
    _LOGGER.info(
        json.dumps(
            {"event": event, **fields},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


__all__ = [
    "CURATED_VISUAL_DOMAIN_NEGATIVE_SOURCE_SCHEMA_VERSION",
    "NEGATIVE_CATEGORIES",
    "NEGATIVE_REVIEW_CONFIDENCE_VALUES",
    "NEGATIVE_REVIEW_STATUSES",
    "NEGATIVE_SOURCE_KINDS",
    "NEGATIVE_TARGET_PRESENCE_VALUES",
    "REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_FILE",
    "REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_REPORT_FILE",
    "REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_REPORT_SCHEMA_VERSION",
    "REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_SCHEMA_VERSION",
    "CuratedVisualDomainNegativeSourceAdapter",
    "compile_curated_visual_domain_negative_manifest",
    "curated_visual_domain_negative_manifest_schema",
    "load_curated_visual_domain_negative_source",
    "publish_curated_visual_domain_negative_manifest",
    "validate_curated_visual_domain_negative_manifest",
    "write_curated_visual_domain_negative_manifest",
]
