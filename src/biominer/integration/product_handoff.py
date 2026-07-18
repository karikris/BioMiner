"""Shared invariants for immutable product handoff manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path, PurePosixPath
import re
from tempfile import NamedTemporaryFile

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.reports.evidence_maturity import EVIDENCE_MATURITY_LABELS


ARTIFACT_AVAILABILITY_STATES = (
    "available",
    "unavailable",
    "withheld",
    "not_applicable",
)

_ARTIFACT_INPUT_FIELDS = frozenset(
    {
        "role",
        "availability",
        "unavailable_reason",
        "relative_path",
        "media_type",
        "schema_version",
        "semantic_fingerprint",
        "sha256",
        "byte_count",
        "row_count",
        "parent_fingerprints",
        "evidence_maturity_label",
    }
)
_NORMALIZED_ARTIFACT_FIELDS = _ARTIFACT_INPUT_FIELDS | frozenset(
    {"scientific_claim_allowed", "producer_repository", "producer_commit"}
)
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def normalize_product_artifacts(
    artifacts: Sequence[Mapping[str, object]],
    *,
    required_roles: Sequence[str],
    producer_repository: str,
    producer_commit: str,
) -> list[dict[str, object]]:
    """Validate and canonically order role-specific handoff artifacts."""

    repository = _required_text(producer_repository, field="producer_repository")
    commit = validate_git_sha(producer_commit, field="producer_commit")
    role_order = tuple(required_roles)
    if not role_order or len(set(role_order)) != len(role_order):
        raise ValueError("required artifact roles must be nonempty and unique")
    by_role: dict[str, Mapping[str, object]] = {}
    for artifact in artifacts:
        unexpected = set(artifact) - _ARTIFACT_INPUT_FIELDS
        missing = _ARTIFACT_INPUT_FIELDS - set(artifact)
        if unexpected or missing:
            raise ValueError(
                "artifact descriptor fields differ from the handoff contract: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        role = _required_text(artifact["role"], field="artifact.role")
        if role in by_role:
            raise ValueError(f"artifact role {role!r} is duplicated")
        by_role[role] = artifact
    if set(by_role) != set(role_order):
        raise ValueError(
            "artifact roles differ from the required contract: "
            f"expected={sorted(role_order)}, actual={sorted(by_role)}"
        )
    return [
        _normalize_artifact(
            by_role[role],
            producer_repository=repository,
            producer_commit=commit,
        )
        for role in role_order
    ]


def finalize_product_manifest(
    body: Mapping[str, object],
    *,
    identity_prefix: str,
) -> dict[str, object]:
    """Add a content-derived identity and fingerprint to a manifest body."""

    prefix = _required_text(identity_prefix, field="identity_prefix")
    if "handoff_id" in body or "manifest_fingerprint" in body:
        raise ValueError("manifest body must not supply derived identity fields")
    normalized = dict(body)
    handoff_id = prefix + canonical_semantic_fingerprint(normalized).removeprefix(
        "sha256:"
    )
    manifest = {**normalized, "handoff_id": handoff_id}
    manifest["manifest_fingerprint"] = canonical_semantic_fingerprint(manifest)
    return manifest


def validate_normalized_product_artifacts(
    artifacts: Sequence[Mapping[str, object]],
    *,
    required_roles: Sequence[str],
    producer_repository: str,
    producer_commit: str,
) -> list[dict[str, object]]:
    """Revalidate stored artifact descriptors rather than trusting their hash."""

    inputs: list[dict[str, object]] = []
    for artifact in artifacts:
        if set(artifact) != _NORMALIZED_ARTIFACT_FIELDS:
            raise ValueError("stored artifact descriptor fields differ")
        if artifact["scientific_claim_allowed"] is not False:
            raise ValueError("product artifact cannot authorize a scientific claim")
        if artifact["producer_repository"] != producer_repository:
            raise ValueError("stored artifact producer repository differs")
        if artifact["producer_commit"] != producer_commit:
            raise ValueError("stored artifact producer commit differs")
        inputs.append({field: artifact[field] for field in _ARTIFACT_INPUT_FIELDS})
    normalized = normalize_product_artifacts(
        inputs,
        required_roles=required_roles,
        producer_repository=producer_repository,
        producer_commit=producer_commit,
    )
    if list(artifacts) != normalized:
        raise ValueError("stored artifact descriptors are not canonical")
    return normalized


def validate_product_manifest_identity(
    manifest: Mapping[str, object],
    *,
    identity_prefix: str,
) -> None:
    """Reject mutation of either derived product-manifest identity."""

    body = dict(manifest)
    fingerprint = body.pop("manifest_fingerprint", None)
    handoff_id = body.pop("handoff_id", None)
    expected = finalize_product_manifest(body, identity_prefix=identity_prefix)
    if handoff_id != expected["handoff_id"]:
        raise ValueError("product handoff identity mismatch")
    if fingerprint != expected["manifest_fingerprint"]:
        raise ValueError("product handoff fingerprint mismatch")


def write_product_manifest(
    manifest: Mapping[str, object],
    path: str | Path,
) -> Path:
    """Atomically write canonical, human-readable manifest JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
        temporary_path.replace(destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def validate_git_sha(value: object, *, field: str) -> str:
    normalized = str(value).strip().casefold()
    if _GIT_SHA_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be a full lowercase Git SHA")
    return normalized


def validate_fingerprint(value: object, *, field: str) -> str:
    normalized = str(value).strip().casefold()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be a sha256: fingerprint")
    return normalized


def _normalize_artifact(
    artifact: Mapping[str, object],
    *,
    producer_repository: str,
    producer_commit: str,
) -> dict[str, object]:
    role = _required_text(artifact["role"], field="artifact.role")
    availability = _required_text(
        artifact["availability"], field=f"artifact[{role}].availability"
    )
    if availability not in ARTIFACT_AVAILABILITY_STATES:
        raise ValueError(f"artifact {role!r} has unsupported availability")
    media_type = _required_text(
        artifact["media_type"], field=f"artifact[{role}].media_type"
    )
    schema_version = _required_text(
        artifact["schema_version"], field=f"artifact[{role}].schema_version"
    )
    maturity = artifact["evidence_maturity_label"]
    if maturity is not None and maturity not in EVIDENCE_MATURITY_LABELS:
        raise ValueError(f"artifact {role!r} has unsupported evidence maturity")
    parents = artifact["parent_fingerprints"]
    if not isinstance(parents, Sequence) or isinstance(parents, (str, bytes)):
        raise ValueError(f"artifact {role!r} parent_fingerprints must be an array")
    parent_fingerprints = sorted(
        validate_fingerprint(value, field=f"artifact[{role}].parent_fingerprints")
        for value in parents
    )
    if len(parent_fingerprints) != len(set(parent_fingerprints)):
        raise ValueError(f"artifact {role!r} parent fingerprints repeat")

    reason = artifact["unavailable_reason"]
    relative_path = artifact["relative_path"]
    semantic_fingerprint = artifact["semantic_fingerprint"]
    physical_sha256 = artifact["sha256"]
    byte_count = artifact["byte_count"]
    row_count = artifact["row_count"]
    if availability == "available":
        if reason is not None:
            raise ValueError(f"available artifact {role!r} cannot have a reason")
        normalized_path = _relative_posix_path(relative_path, role=role)
        semantic = validate_fingerprint(
            semantic_fingerprint, field=f"artifact[{role}].semantic_fingerprint"
        )
        physical = validate_fingerprint(
            physical_sha256, field=f"artifact[{role}].sha256"
        )
        bytes_value = _nonnegative_int(byte_count, field=f"artifact[{role}].byte_count")
        if bytes_value == 0:
            raise ValueError(f"available artifact {role!r} must contain bytes")
        rows_value = _nonnegative_int(row_count, field=f"artifact[{role}].row_count")
    else:
        normalized_reason = _required_text(
            reason, field=f"artifact[{role}].unavailable_reason"
        )
        if any(
            value is not None
            for value in (
                relative_path,
                semantic_fingerprint,
                physical_sha256,
                byte_count,
                row_count,
            )
        ):
            raise ValueError(
                f"non-available artifact {role!r} cannot claim physical identity"
            )
        normalized_path = None
        semantic = None
        physical = None
        bytes_value = None
        rows_value = None
        reason = normalized_reason
    return {
        "role": role,
        "availability": availability,
        "unavailable_reason": reason,
        "relative_path": normalized_path,
        "media_type": media_type,
        "schema_version": schema_version,
        "semantic_fingerprint": semantic,
        "sha256": physical,
        "byte_count": bytes_value,
        "row_count": rows_value,
        "parent_fingerprints": parent_fingerprints,
        "evidence_maturity_label": maturity,
        "scientific_claim_allowed": False,
        "producer_repository": producer_repository,
        "producer_commit": producer_commit,
    }


def _relative_posix_path(value: object, *, role: str) -> str:
    text = _required_text(value, field=f"artifact[{role}].relative_path")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ValueError(f"artifact {role!r} path must be canonical and relative")
    return text


def _required_text(value: object, *, field: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


__all__ = [
    "ARTIFACT_AVAILABILITY_STATES",
    "finalize_product_manifest",
    "normalize_product_artifacts",
    "validate_normalized_product_artifacts",
    "validate_fingerprint",
    "validate_git_sha",
    "validate_product_manifest_identity",
    "write_product_manifest",
]
