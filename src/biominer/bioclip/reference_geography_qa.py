"""Cross-artifact QA and manifest for the geographic reference index."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import json
from pathlib import Path
import re
import tempfile

import polars as pl

from biominer.bioclip.geographic_reference_neighbours import (
    GEOGRAPHIC_REFERENCE_NEIGHBOURS_FILE,
    GEOGRAPHIC_REFERENCE_NEIGHBOURS_SCHEMA_VERSION,
    geographic_reference_neighbours_artifact_fingerprint,
    geographic_reference_neighbours_schema,
    validate_geographic_reference_neighbours,
)
from biominer.bioclip.global_reference_anchors import (
    GLOBAL_REFERENCE_ANCHORS_FILE,
    GLOBAL_REFERENCE_ANCHORS_SCHEMA_VERSION,
    global_reference_anchors_artifact_fingerprint,
    global_reference_anchors_schema,
    validate_global_reference_anchors,
)
from biominer.bioclip.reference_geography_index import (
    REFERENCE_GEOGRAPHY_INDEX_FILE,
    REFERENCE_GEOGRAPHY_INDEX_SCHEMA_VERSION,
    reference_geography_index_artifact_fingerprint,
    reference_geography_index_schema,
    validate_reference_geography_index,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.normalized_geography import (
    NORMALIZED_REFERENCE_GEOGRAPHY_FILE,
    NORMALIZED_REFERENCE_GEOGRAPHY_SCHEMA_VERSION,
    normalized_reference_geography_artifact_fingerprint,
    normalized_reference_geography_schema,
    validate_normalized_reference_geography,
)


REFERENCE_GEOGRAPHY_INDEX_MANIFEST_SCHEMA_VERSION = (
    "reference-geography-index-manifest-v1.0.0"
)
REFERENCE_GEOGRAPHY_INDEX_MANIFEST_FILE = "reference_geography_index_manifest.json"
REFERENCE_GEOGRAPHY_QA_POLICY_VERSION = "reference-geography-index-qa-v1.0.0"

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_SEVERITIES = frozenset({"fatal", "warning"})
_ARTIFACT_CONTRACTS = {
    "normalized_reference_geography": (
        NORMALIZED_REFERENCE_GEOGRAPHY_FILE,
        NORMALIZED_REFERENCE_GEOGRAPHY_SCHEMA_VERSION,
        normalized_reference_geography_schema,
        validate_normalized_reference_geography,
        normalized_reference_geography_artifact_fingerprint,
    ),
    "reference_geography_index": (
        REFERENCE_GEOGRAPHY_INDEX_FILE,
        REFERENCE_GEOGRAPHY_INDEX_SCHEMA_VERSION,
        reference_geography_index_schema,
        validate_reference_geography_index,
        reference_geography_index_artifact_fingerprint,
    ),
    "global_reference_anchors": (
        GLOBAL_REFERENCE_ANCHORS_FILE,
        GLOBAL_REFERENCE_ANCHORS_SCHEMA_VERSION,
        global_reference_anchors_schema,
        validate_global_reference_anchors,
        global_reference_anchors_artifact_fingerprint,
    ),
    "geographic_reference_neighbours": (
        GEOGRAPHIC_REFERENCE_NEIGHBOURS_FILE,
        GEOGRAPHIC_REFERENCE_NEIGHBOURS_SCHEMA_VERSION,
        geographic_reference_neighbours_schema,
        validate_geographic_reference_neighbours,
        geographic_reference_neighbours_artifact_fingerprint,
    ),
}


def build_reference_geography_index_manifest(
    reference_geography_index: pl.DataFrame,
    normalized_reference_geography: pl.DataFrame,
    global_reference_anchors: pl.DataFrame,
    geographic_reference_neighbours: pl.DataFrame,
    *,
    producer_git_sha: str,
    physical_sha256s: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Audit all Phase 2 reference artifacts and return a reproducible manifest."""

    producer = _git_sha(producer_git_sha)
    supplied_checksums = _physical_checksums(physical_sha256s)
    frames = {
        "normalized_reference_geography": normalized_reference_geography,
        "reference_geography_index": reference_geography_index,
        "global_reference_anchors": global_reference_anchors,
        "geographic_reference_neighbours": geographic_reference_neighbours,
    }
    findings: list[dict[str, str]] = []
    artifacts: dict[str, dict[str, object]] = {}
    contract_validity: dict[str, bool] = {}
    for name, frame in frames.items():
        record, valid, artifact_findings = _artifact_record(
            name,
            frame,
            physical_sha256=supplied_checksums.get(name),
        )
        artifacts[name] = record
        contract_validity[name] = valid
        findings.extend(artifact_findings)

    metrics = _base_metrics(frames)
    if all(contract_validity.values()):
        cross_metrics, cross_findings = _cross_artifact_audit(
            reference_geography_index=reference_geography_index,
            normalized_reference_geography=normalized_reference_geography,
            global_reference_anchors=global_reference_anchors,
            geographic_reference_neighbours=geographic_reference_neighbours,
        )
        metrics.update(cross_metrics)
        findings.extend(cross_findings)
    else:
        metrics["cross_artifact_audit_status"] = "unavailable_invalid_contract"

    findings = sorted(
        findings,
        key=lambda row: (
            row["severity"],
            row["code"],
            row["subject"],
            row["detail"],
        ),
    )
    fatal_count = sum(row["severity"] == "fatal" for row in findings)
    warning_count = sum(row["severity"] == "warning" for row in findings)
    policy_fingerprint = _qa_policy_fingerprint()
    payload: dict[str, object] = {
        "schema_version": REFERENCE_GEOGRAPHY_INDEX_MANIFEST_SCHEMA_VERSION,
        "qa_policy_version": REFERENCE_GEOGRAPHY_QA_POLICY_VERSION,
        "qa_policy_fingerprint": policy_fingerprint,
        "producer_git_sha": producer,
        "qa_status": "failed" if fatal_count else "passed",
        "fatal_finding_count": fatal_count,
        "warning_finding_count": warning_count,
        "findings": findings,
        "artifacts": artifacts,
        "source_snapshot_versions": sorted(
            normalized_reference_geography["source_snapshot_version"].unique().to_list()
        )
        if contract_validity["normalized_reference_geography"]
        else [],
        "metrics": metrics,
        "physical_checksum_status": (
            "complete"
            if set(supplied_checksums) == set(_ARTIFACT_CONTRACTS)
            else "unavailable_not_supplied"
        ),
    }
    payload["manifest_fingerprint"] = canonical_semantic_fingerprint(payload)
    validate_reference_geography_index_manifest(payload)
    return payload


def validate_reference_geography_index_manifest(
    manifest: Mapping[str, object],
) -> None:
    """Reject incomplete, noncanonical, or self-inconsistent QA manifests."""

    if not isinstance(manifest, Mapping):
        raise TypeError("reference geography index manifest must be a mapping")
    expected = {
        "schema_version",
        "qa_policy_version",
        "qa_policy_fingerprint",
        "producer_git_sha",
        "qa_status",
        "fatal_finding_count",
        "warning_finding_count",
        "findings",
        "artifacts",
        "source_snapshot_versions",
        "metrics",
        "physical_checksum_status",
        "manifest_fingerprint",
    }
    if set(manifest) != expected:
        raise ValueError("reference geography index manifest fields mismatch")
    if manifest["schema_version"] != REFERENCE_GEOGRAPHY_INDEX_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported reference geography index manifest schema")
    if manifest["qa_policy_version"] != REFERENCE_GEOGRAPHY_QA_POLICY_VERSION:
        raise ValueError("unsupported reference geography QA policy")
    for field in ("qa_policy_fingerprint", "manifest_fingerprint"):
        if not _SHA256_PATTERN.fullmatch(str(manifest[field])):
            raise ValueError(f"{field} is not a canonical SHA-256 fingerprint")
    if manifest["qa_policy_fingerprint"] != _qa_policy_fingerprint():
        raise ValueError("reference geography QA policy fingerprint drifted")
    _git_sha(manifest["producer_git_sha"])
    if manifest["qa_status"] not in {"passed", "failed"}:
        raise ValueError("reference geography manifest qa_status is invalid")
    findings = manifest["findings"]
    if not isinstance(findings, list):
        raise TypeError("reference geography manifest findings must be a list")
    _validate_findings(findings)
    fatal_count = sum(row["severity"] == "fatal" for row in findings)
    warning_count = sum(row["severity"] == "warning" for row in findings)
    if manifest["fatal_finding_count"] != fatal_count:
        raise ValueError("reference geography manifest fatal count drifted")
    if manifest["warning_finding_count"] != warning_count:
        raise ValueError("reference geography manifest warning count drifted")
    if manifest["qa_status"] != ("failed" if fatal_count else "passed"):
        raise ValueError(
            "reference geography manifest QA status conflicts with findings"
        )
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(_ARTIFACT_CONTRACTS):
        raise ValueError("reference geography manifest artifact set mismatch")
    for name, record in artifacts.items():
        _validate_artifact_record(str(name), record)
    if manifest["qa_status"] == "passed" and any(
        record["semantic_fingerprint"] is None for record in artifacts.values()
    ):
        raise ValueError("passed reference geography manifest lacks artifact identity")
    snapshots = manifest["source_snapshot_versions"]
    if (
        not isinstance(snapshots, list)
        or snapshots != sorted(set(snapshots))
        or any(not isinstance(item, str) or not item for item in snapshots)
    ):
        raise ValueError("source_snapshot_versions are not canonical")
    if not isinstance(manifest["metrics"], Mapping):
        raise TypeError("reference geography manifest metrics must be a mapping")
    if manifest["physical_checksum_status"] not in {
        "complete",
        "unavailable_not_supplied",
    }:
        raise ValueError("reference geography physical checksum status is invalid")
    complete_physical = all(
        record["physical_checksum_status"] == "available"
        for record in artifacts.values()
    )
    if manifest["physical_checksum_status"] != (
        "complete" if complete_physical else "unavailable_not_supplied"
    ):
        raise ValueError("reference geography physical checksum summary drifted")
    payload = dict(manifest)
    fingerprint = payload.pop("manifest_fingerprint")
    if fingerprint != canonical_semantic_fingerprint(payload):
        raise ValueError("reference geography manifest fingerprint mismatch")


def write_reference_geography_index_manifest(
    manifest: Mapping[str, object],
    output: str | Path,
) -> Path:
    """Validate and atomically write ``reference_geography_index_manifest.json``."""

    validate_reference_geography_index_manifest(manifest)
    destination = Path(output)
    if destination.suffix.casefold() != ".json":
        destination /= REFERENCE_GEOGRAPHY_INDEX_MANIFEST_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(manifest), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
    temporary.replace(destination)
    return destination


def _artifact_record(
    name: str,
    frame: object,
    *,
    physical_sha256: str | None,
) -> tuple[dict[str, object], bool, list[dict[str, str]]]:
    filename, schema_version, schema_factory, validator, fingerprinter = (
        _ARTIFACT_CONTRACTS[name]
    )
    findings: list[dict[str, str]] = []
    if not isinstance(frame, pl.DataFrame):
        findings.append(
            _finding("fatal", "artifact_contract_invalid", name, "not_a_dataframe")
        )
        return (
            _artifact_payload(
                filename,
                schema_version,
                row_count=None,
                semantic_fingerprint=None,
                physical_sha256=physical_sha256,
            ),
            False,
            findings,
        )
    schema_valid = frame.schema == schema_factory()
    contract_error: str | None = None
    if schema_valid:
        try:
            validator(frame)
        except (TypeError, ValueError) as exc:
            contract_error = str(exc)
    else:
        contract_error = "schema_mismatch"
    valid = contract_error is None
    semantic_fingerprint: str | None = None
    if valid:
        semantic_fingerprint = fingerprinter(frame)
    else:
        findings.append(
            _finding(
                "fatal",
                "artifact_contract_invalid",
                name,
                contract_error or "unknown_contract_failure",
            )
        )
    return (
        _artifact_payload(
            filename,
            schema_version,
            row_count=frame.height,
            semantic_fingerprint=semantic_fingerprint,
            physical_sha256=physical_sha256,
        ),
        valid,
        findings,
    )


def _artifact_payload(
    filename: str,
    schema_version: str,
    *,
    row_count: int | None,
    semantic_fingerprint: str | None,
    physical_sha256: str | None,
) -> dict[str, object]:
    return {
        "file": filename,
        "schema_version": schema_version,
        "row_count": row_count,
        "semantic_fingerprint": semantic_fingerprint,
        "physical_sha256": physical_sha256,
        "physical_checksum_status": (
            "available" if physical_sha256 is not None else "unavailable_not_supplied"
        ),
    }


def _base_metrics(frames: Mapping[str, object]) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name, frame in frames.items():
        metrics[f"{name}_row_count"] = (
            frame.height if isinstance(frame, pl.DataFrame) else None
        )
    return metrics


def _cross_artifact_audit(
    *,
    reference_geography_index: pl.DataFrame,
    normalized_reference_geography: pl.DataFrame,
    global_reference_anchors: pl.DataFrame,
    geographic_reference_neighbours: pl.DataFrame,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    findings: list[dict[str, str]] = []
    index_fingerprint = reference_geography_index_artifact_fingerprint(
        reference_geography_index
    )
    normalized_fingerprint = normalized_reference_geography_artifact_fingerprint(
        normalized_reference_geography
    )
    anchors_fingerprint = global_reference_anchors_artifact_fingerprint(
        global_reference_anchors
    )
    index_rows = set(reference_geography_index["row_fingerprint"].to_list())
    index_observations = set(
        reference_geography_index["reference_observation_id"].to_list()
    )
    normalized_observations = set(
        normalized_reference_geography["reference_observation_id"].to_list()
    )
    missing_geography = sorted(index_observations - normalized_observations)
    if missing_geography:
        findings.append(
            _finding(
                "fatal",
                "normalized_geography_missing_observations",
                "reference_geography_index",
                f"count={len(missing_geography)}",
            )
        )

    anchor_rows = set(
        global_reference_anchors["reference_geography_row_fingerprint"].to_list()
    )
    unknown_anchor_rows = anchor_rows - index_rows
    if unknown_anchor_rows:
        findings.append(
            _finding(
                "fatal",
                "global_anchor_unknown_index_rows",
                "global_reference_anchors",
                f"count={len(unknown_anchor_rows)}",
            )
        )
    if not global_reference_anchors.is_empty() and set(
        global_reference_anchors["reference_geography_index_fingerprint"].to_list()
    ) != {index_fingerprint}:
        findings.append(
            _finding(
                "fatal",
                "global_anchor_index_lineage_mismatch",
                "global_reference_anchors",
                "parent_fingerprint_differs",
            )
        )

    neighbour_parent_rows = set(
        geographic_reference_neighbours["reference_geography_row_fingerprint"].to_list()
    )
    unknown_neighbour_rows = neighbour_parent_rows - index_rows
    if unknown_neighbour_rows:
        findings.append(
            _finding(
                "fatal",
                "neighbour_unknown_index_rows",
                "geographic_reference_neighbours",
                f"count={len(unknown_neighbour_rows)}",
            )
        )
    _audit_neighbour_lineage(
        geographic_reference_neighbours,
        index_fingerprint=index_fingerprint,
        normalized_fingerprint=normalized_fingerprint,
        anchors_fingerprint=anchors_fingerprint,
        findings=findings,
    )
    cross_metrics = _audit_memberships(
        reference_geography_index=reference_geography_index,
        normalized_reference_geography=normalized_reference_geography,
        global_reference_anchors=global_reference_anchors,
        geographic_reference_neighbours=geographic_reference_neighbours,
        findings=findings,
    )
    metrics: dict[str, object] = {
        "cross_artifact_audit_status": "completed",
        "reference_media_count": reference_geography_index[
            "reference_media_id"
        ].n_unique(),
        "reference_embedding_count": reference_geography_index[
            "embedding_fingerprint"
        ].n_unique(),
        "reference_observation_count": len(index_observations),
        "reference_duplicate_group_count": reference_geography_index[
            "duplicate_group_id"
        ].n_unique(),
        "normalized_observations_not_indexed_count": len(
            normalized_observations - index_observations
        ),
        "global_anchor_observation_inflation_count": (
            global_reference_anchors.height
            - global_reference_anchors["reference_observation_id"].n_unique()
        ),
        "global_anchor_duplicate_group_inflation_count": (
            global_reference_anchors.height
            - global_reference_anchors["duplicate_group_id"].n_unique()
        ),
        "global_anchor_photographer_count": global_reference_anchors["observer_id_hash"]
        .drop_nulls()
        .n_unique(),
        "global_anchor_country_count": global_reference_anchors["country_code"]
        .drop_nulls()
        .n_unique(),
        "global_anchor_total_shortfall": sum(
            group["anchor_shortfall"][0]
            for group in global_reference_anchors.partition_by(
                ["accepted_taxon_key", "route"]
            )
        ),
        **cross_metrics,
    }
    return metrics, findings


def _audit_neighbour_lineage(
    neighbours: pl.DataFrame,
    *,
    index_fingerprint: str,
    normalized_fingerprint: str,
    anchors_fingerprint: str,
    findings: list[dict[str, str]],
) -> None:
    expected = {
        "reference_geography_index_fingerprint": index_fingerprint,
        "normalized_reference_geography_fingerprint": normalized_fingerprint,
        "global_reference_anchors_fingerprint": anchors_fingerprint,
    }
    for field, fingerprint in expected.items():
        actual = set(neighbours[field].to_list())
        if neighbours.is_empty():
            continue
        if actual != {fingerprint}:
            findings.append(
                _finding(
                    "fatal",
                    "neighbour_input_lineage_mismatch",
                    field,
                    "parent_fingerprint_differs",
                )
            )


def _audit_memberships(
    *,
    reference_geography_index: pl.DataFrame,
    normalized_reference_geography: pl.DataFrame,
    global_reference_anchors: pl.DataFrame,
    geographic_reference_neighbours: pl.DataFrame,
    findings: list[dict[str, str]],
) -> dict[str, object]:
    memberships_by_parent: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in geographic_reference_neighbours.iter_rows(named=True):
        memberships_by_parent[str(row["reference_geography_row_fingerprint"])].append(
            row
        )
    geography_by_observation = {
        str(row["reference_observation_id"]): row
        for row in normalized_reference_geography.iter_rows(named=True)
    }
    anchor_rows = set(
        global_reference_anchors["reference_geography_row_fingerprint"].to_list()
    )
    false_local_count = 0
    precision_violation_count = 0
    missing_exact_count = 0
    missing_fallback_count = 0
    unexpected_global_count = 0
    for index_row in reference_geography_index.iter_rows(named=True):
        parent = str(index_row["row_fingerprint"])
        memberships = memberships_by_parent.get(parent, [])
        scopes = Counter(str(row["lookup_scope"]) for row in memberships)
        cell_rows = [
            row for row in memberships if row["lookup_cell_resolution"] is not None
        ]
        if index_row["local_anchor_eligible"]:
            if scopes["exact_supported_cell"] != 1:
                missing_exact_count += 1
                findings.append(
                    _finding(
                        "fatal",
                        "local_reference_exact_membership_missing",
                        parent,
                        f"count={scopes['exact_supported_cell']}",
                    )
                )
        elif cell_rows:
            false_local_count += len(cell_rows)
            findings.append(
                _finding(
                    "fatal",
                    "false_local_membership",
                    parent,
                    f"count={len(cell_rows)}",
                )
            )
        for row in cell_rows:
            if int(row["lookup_cell_resolution"]) > int(
                row["supported_cell_resolution"]
            ):
                precision_violation_count += 1
        geography = geography_by_observation.get(
            str(index_row["reference_observation_id"])
        )
        if geography is not None:
            expected_named = {
                scope
                for scope, field in (
                    ("bioregion", "bioregion"),
                    ("country", "country_code"),
                    ("continent", "continent_code"),
                )
                if geography[field] is not None
            }
            missing_named = sorted(
                scope for scope in expected_named if scopes[scope] != 1
            )
            if missing_named:
                missing_fallback_count += len(missing_named)
                findings.append(
                    _finding(
                        "fatal",
                        "named_fallback_membership_missing",
                        parent,
                        ",".join(missing_named),
                    )
                )
        expected_global = parent in anchor_rows
        if expected_global and scopes["global"] != 1:
            missing_fallback_count += 1
            findings.append(
                _finding(
                    "fatal",
                    "global_fallback_membership_missing",
                    parent,
                    f"count={scopes['global']}",
                )
            )
        if not expected_global and scopes["global"]:
            unexpected_global_count += scopes["global"]
            findings.append(
                _finding(
                    "fatal",
                    "global_fallback_membership_unselected",
                    parent,
                    f"count={scopes['global']}",
                )
            )
        wrong_anchor_flags = sum(
            bool(row["is_global_anchor"]) != expected_global for row in memberships
        )
        if wrong_anchor_flags:
            findings.append(
                _finding(
                    "fatal",
                    "global_anchor_flag_mismatch",
                    parent,
                    f"count={wrong_anchor_flags}",
                )
            )
    if precision_violation_count:
        findings.append(
            _finding(
                "fatal",
                "reference_precision_leakage",
                "geographic_reference_neighbours",
                f"count={precision_violation_count}",
            )
        )
    scope_counts = Counter(geographic_reference_neighbours["lookup_scope"].to_list())
    bucket_observation_counts: dict[tuple[str, str, str, str], Counter[str]] = (
        defaultdict(Counter)
    )
    for row in geographic_reference_neighbours.iter_rows(named=True):
        bucket = (
            str(row["lookup_scope"]),
            str(row["lookup_key"]),
            str(row["accepted_taxon_key"]),
            str(row["route"]),
        )
        bucket_observation_counts[bucket][str(row["reference_observation_id"])] += 1
    max_rows_per_observation = max(
        (
            max(counts.values())
            for counts in bucket_observation_counts.values()
            if counts
        ),
        default=0,
    )
    return {
        "neighbour_membership_count": geographic_reference_neighbours.height,
        "neighbour_scope_counts": {
            scope: scope_counts.get(scope, 0) for scope in sorted(scope_counts)
        },
        "false_local_membership_count": false_local_count,
        "precision_violation_count": precision_violation_count,
        "missing_exact_membership_count": missing_exact_count,
        "missing_fallback_membership_count": missing_fallback_count,
        "unexpected_global_membership_count": unexpected_global_count,
        "maximum_embedding_rows_per_observation_lookup": max_rows_per_observation,
        "lookup_independence_counting_unit": "distinct_reference_observation_id",
    }


def _finding(
    severity: str,
    code: str,
    subject: str,
    detail: str,
) -> dict[str, str]:
    return {
        "severity": severity,
        "code": code,
        "subject": subject,
        "detail": detail,
    }


def _qa_policy_fingerprint() -> str:
    return canonical_semantic_fingerprint(
        {
            "version": REFERENCE_GEOGRAPHY_QA_POLICY_VERSION,
            "required_artifacts": sorted(_ARTIFACT_CONTRACTS),
            "fatal_checks": [
                "artifact_contract",
                "input_lineage",
                "normalized_geography_coverage",
                "global_anchor_independence",
                "lookup_membership_grain",
                "precision_aware_cells",
                "false_local_assignment",
                "global_fallback_parity",
                "required_fallback_provenance",
            ],
            "embedding_rows_are_not_independent_observations": True,
        }
    )


def _validate_findings(findings: list[object]) -> None:
    expected_fields = {"severity", "code", "subject", "detail"}
    previous: tuple[str, str, str, str] | None = None
    for finding in findings:
        if not isinstance(finding, Mapping) or set(finding) != expected_fields:
            raise ValueError("reference geography finding fields mismatch")
        values = tuple(str(finding[field]) for field in expected_fields)
        if any(not value for value in values):
            raise ValueError("reference geography finding contains blank text")
        if finding["severity"] not in _SEVERITIES:
            raise ValueError("reference geography finding severity is invalid")
        current = (
            str(finding["severity"]),
            str(finding["code"]),
            str(finding["subject"]),
            str(finding["detail"]),
        )
        if previous is not None and current < previous:
            raise ValueError("reference geography findings are not sorted")
        previous = current


def _validate_artifact_record(name: str, record: object) -> None:
    if not isinstance(record, Mapping):
        raise TypeError("reference geography artifact record must be a mapping")
    expected_fields = {
        "file",
        "schema_version",
        "row_count",
        "semantic_fingerprint",
        "physical_sha256",
        "physical_checksum_status",
    }
    if set(record) != expected_fields:
        raise ValueError("reference geography artifact record fields mismatch")
    filename, schema_version, *_ = _ARTIFACT_CONTRACTS[name]
    if record["file"] != filename or record["schema_version"] != schema_version:
        raise ValueError("reference geography artifact record contract mismatch")
    row_count = record["row_count"]
    if row_count is not None and (
        isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0
    ):
        raise ValueError("reference geography artifact row_count is invalid")
    for field in ("semantic_fingerprint", "physical_sha256"):
        value = record[field]
        if value is not None and not _SHA256_PATTERN.fullmatch(str(value)):
            raise ValueError(f"reference geography artifact {field} is invalid")
    expected_status = (
        "available"
        if record["physical_sha256"] is not None
        else "unavailable_not_supplied"
    )
    if record["physical_checksum_status"] != expected_status:
        raise ValueError("reference geography artifact checksum status drifted")


def _physical_checksums(value: Mapping[str, str] | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("physical_sha256s must be a mapping")
    unknown = set(value) - set(_ARTIFACT_CONTRACTS)
    if unknown:
        raise ValueError(
            f"physical_sha256s contain unknown artifacts: {sorted(unknown)}"
        )
    output: dict[str, str] = {}
    for name, fingerprint in value.items():
        if not _SHA256_PATTERN.fullmatch(str(fingerprint)):
            raise ValueError(f"physical checksum for {name} is invalid")
        output[str(name)] = str(fingerprint)
    return output


def _git_sha(value: object) -> str:
    if not isinstance(value, str) or not _GIT_SHA_PATTERN.fullmatch(value):
        raise ValueError("producer_git_sha must be a lowercase 40-character Git SHA")
    return value


__all__ = [
    "REFERENCE_GEOGRAPHY_INDEX_MANIFEST_FILE",
    "REFERENCE_GEOGRAPHY_INDEX_MANIFEST_SCHEMA_VERSION",
    "REFERENCE_GEOGRAPHY_QA_POLICY_VERSION",
    "build_reference_geography_index_manifest",
    "validate_reference_geography_index_manifest",
    "write_reference_geography_index_manifest",
]
