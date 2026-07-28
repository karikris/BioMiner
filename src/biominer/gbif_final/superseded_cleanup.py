from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence
from uuid import uuid4

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.gbif_final.publication_audit import (
    validate_publication_audit,
)


CLEANUP_INTENT_VERSION = "gbif-final-superseded-cleanup-intent/v1"
CLEANUP_PROGRESS_VERSION = "gbif-final-superseded-cleanup-progress/v1"
CLEANUP_MANIFEST_VERSION = "gbif-final-superseded-cleanup/v1"

SUPERSEDED_RELATIVE_PATHS = (
    "data/derived/gbif_media_database/v1",
    "data/derived/gbif_media_database/v2",
    "data/derived/gbif_media_temporal/v1",
    "data/derived/gbif_flickr_keyword_registry/v1",
    "data/derived/gbif_flickr_keyword_registry/v2",
    "data/reference/gbif_global_papilionoidea_occurrence_multimedia_parquet",
    "data/reference/"
    "gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_parquet",
    "data/reference/"
    "gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_"
    "completeness_ge_5pct_parquet",
    "data/reference/"
    "gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_"
    "completeness_ge_5pct_verification_grouped_parquet",
    "data/reference/"
    "gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_"
    "completeness_ge_5pct_verification_identified_as_accepted_parquet",
    "data/reference/"
    "gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_"
    "completeness_ge_5pct_verification_normalized_parquet",
    "data/reference/"
    "gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_"
    "completeness_ge_5pct_verification_normalized_identified_by_present_"
    "parquet",
    "data/reference/"
    "gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_"
    "completeness_ge_5pct_verification_normalized_identified_by_or_"
    "accepted_parquet",
)

PROTECTED_RELATIVE_PATHS = (
    "data/derived/gbif_media_database/v3",
    "data/derived/gbif_media_database/v4",
    "data/reference/"
    "gbif_global_papilionoidea_occurrence_multimedia_year_ge_1960_"
    "completeness_ge_5pct_verification_normalized_identified_by_or_"
    "accepted_rights_filtered_parquet",
    "data/reference/gbif_global_papilionoidea_parquet",
    "data/reference/gbif-global-papilionoidea-download-clean.zip",
    "data/derived/gbif_media_audit/v1",
    "data/state/gbif-media-url-resolution",
    "data/state/gbif-media-url-full-v1.sqlite",
)

INTENT_FILENAME = "intent.json"
PROGRESS_FILENAME = "progress.json"
MANIFEST_FILENAME = "manifest.json"


def plan_superseded_cleanup(
    *,
    repository_root: str | Path,
    publication_audit_directory: str | Path,
    state_directory: str | Path,
) -> dict[str, Any]:
    """Build a checksum-bound dry-run plan for the exact obsolete allowlist."""

    repository = Path(repository_root).resolve()
    audit = Path(publication_audit_directory).resolve()
    state = Path(state_directory).resolve()
    _require_directory(repository, "repository root")
    _require_no_overlap(
        state,
        (repository, audit),
        context="cleanup state",
        allow_ancestor=repository,
    )
    audit_manifest = validate_publication_audit(
        audit,
        repository_root=repository,
        require_dependencies=True,
    )
    primary = audit_manifest["primary_publication"]
    primary_manifest = Path(primary["manifest_path"]).resolve()
    final_artifact = Path(
        primary["final_artifact"]["path"]
    ).resolve()
    dynamic_protected = (
        audit,
        primary_manifest.parent,
        final_artifact,
    )
    protected = _protected_paths(repository, dynamic_protected)
    targets = [
        _exact_existing_path(repository, relative)
        for relative in SUPERSEDED_RELATIVE_PATHS
    ]
    _require_no_overlap(
        state,
        (*targets, *dynamic_protected),
        context="cleanup state",
        allow_ancestor=repository,
    )
    _validate_targets(targets, protected, repository)

    target_rows: list[dict[str, object]] = []
    all_files: list[dict[str, object]] = []
    all_directories: list[str] = []
    for relative, target in zip(
        SUPERSEDED_RELATIVE_PATHS,
        targets,
        strict=True,
    ):
        files, directories = _inventory_tree(repository, target)
        target_rows.append(
            {
                "relative_path": relative,
                "file_count": len(files),
                "physical_bytes": sum(
                    int(item["physical_bytes"]) for item in files
                ),
                "tree_fingerprint": canonical_semantic_fingerprint(files),
            }
        )
        all_files.extend(files)
        all_directories.extend(directories)
    all_files.sort(key=lambda item: str(item["relative_path"]))
    all_directories = sorted(
        set(all_directories),
        key=lambda value: (len(Path(value).parts), value),
        reverse=True,
    )
    body: dict[str, Any] = {
        "schema_version": CLEANUP_INTENT_VERSION,
        "generated_at": _timestamp(),
        "repository_root": str(repository),
        "publication_audit_directory": str(audit),
        "publication_audit_manifest_sha256": _sha256(
            audit / MANIFEST_FILENAME
        ),
        "publication_audit_fingerprint": audit_manifest[
            "manifest_fingerprint"
        ],
        "primary_manifest_path": str(primary_manifest),
        "primary_manifest_sha256": primary["manifest_sha256"],
        "final_artifact_path": str(final_artifact),
        "final_artifact_sha256": primary["final_artifact"][
            "physical_sha256"
        ],
        "state_directory": str(state),
        "targets": target_rows,
        "files": all_files,
        "directories": all_directories,
        "protected_paths": [
            _display_path(repository, path) for path in protected
        ],
        "counts": {
            "target_directories": len(targets),
            "files": len(all_files),
            "directories": len(all_directories),
            "physical_bytes": sum(
                int(item["physical_bytes"]) for item in all_files
            ),
        },
        "validation": {
            "publication_audit_revalidated": True,
            "publication_dependencies_revalidated": True,
            "exact_allowlist_used": True,
            "all_targets_exist": True,
            "all_targets_inside_repository": True,
            "targets_contain_no_symlinks": True,
            "targets_do_not_overlap_protected_paths": True,
            "protected_paths_present": all(path.exists() for path in protected),
            "dry_run_only": True,
        },
        "policy": {
            "default_mode": "dry-run",
            "intent_written_before_deletion": True,
            "per_file_checksum_revalidated_before_deletion": True,
            "unexpected_files_fail_closed": True,
            "interrupted_execution_resumable": True,
            "manifest_written_last": True,
        },
    }
    if not all(body["validation"].values()):
        raise RuntimeError(
            f"superseded cleanup plan failed: {body['validation']}"
        )
    body["intent_fingerprint"] = canonical_semantic_fingerprint(body)
    return body


def prepare_superseded_cleanup(
    *,
    repository_root: str | Path,
    publication_audit_directory: str | Path,
    state_directory: str | Path,
) -> dict[str, Any]:
    """Persist immutable intent and restart state without deleting anything."""

    state = Path(state_directory).resolve()
    if state.exists():
        raise FileExistsError(state)
    intent = plan_superseded_cleanup(
        repository_root=repository_root,
        publication_audit_directory=publication_audit_directory,
        state_directory=state,
    )
    staging = state.parent / f".{state.name}.{uuid4().hex}.staging"
    staging.mkdir(parents=True)
    try:
        _write_json_create_only(staging / INTENT_FILENAME, intent)
        progress = {
            "schema_version": CLEANUP_PROGRESS_VERSION,
            "intent_fingerprint": intent["intent_fingerprint"],
            "updated_at": _timestamp(),
            "file_status": {
                str(item["relative_path"]): "pending"
                for item in intent["files"]
            },
            "completed_directories": [],
            "status": "prepared",
        }
        _write_json_atomic(staging / PROGRESS_FILENAME, progress)
        os.replace(staging, state)
        _fsync_directory(state.parent)
        return intent
    except BaseException:
        _remove_empty_staging(staging)
        raise


def execute_superseded_cleanup(
    *,
    repository_root: str | Path,
    publication_audit_directory: str | Path,
    state_directory: str | Path,
) -> dict[str, Any]:
    """Resume or execute the prepared exact cleanup, then seal a manifest."""

    repository = Path(repository_root).resolve()
    audit = Path(publication_audit_directory).resolve()
    state = Path(state_directory).resolve()
    if state.is_symlink() or not state.is_dir():
        raise FileNotFoundError(state)
    manifest_path = state / MANIFEST_FILENAME
    if manifest_path.exists():
        return validate_superseded_cleanup(
            repository_root=repository,
            publication_audit_directory=audit,
            state_directory=state,
        )
    intent = _load_intent(state / INTENT_FILENAME)
    _require_invocation_matches(
        intent,
        repository=repository,
        audit=audit,
        state=state,
    )
    _validate_intent_contract(intent, repository)
    protected = _protected_paths(
        repository,
        (
            audit,
            Path(intent["primary_manifest_path"]).parent,
            Path(intent["final_artifact_path"]),
            state,
        ),
    )
    targets = [
        (repository / relative).resolve()
        for relative in SUPERSEDED_RELATIVE_PATHS
    ]
    _validate_target_contract(targets, protected, repository)
    progress_path = state / PROGRESS_FILENAME
    progress = _load_progress(progress_path, intent)
    cleanup_started = any(
        value != "pending"
        for value in progress["file_status"].values()
    ) or any(
        not (repository / str(item["relative_path"])).exists()
        for item in intent["files"]
    )
    validate_publication_audit(
        audit,
        repository_root=repository,
        require_dependencies=not cleanup_started,
    )
    _require_bound_publication_unchanged(intent)

    progress["status"] = "running"
    progress["updated_at"] = _timestamp()
    _write_json_atomic(progress_path, progress)
    file_by_path = {
        str(item["relative_path"]): item for item in intent["files"]
    }
    for relative in sorted(file_by_path):
        expected = file_by_path[relative]
        current_status = progress["file_status"][relative]
        path = repository / relative
        if current_status in {"deleted", "already_absent_after_intent"}:
            if path.exists() or path.is_symlink():
                raise RuntimeError(
                    f"completed cleanup file unexpectedly exists: {path}"
                )
            continue
        progress["file_status"][relative] = "deleting"
        progress["updated_at"] = _timestamp()
        _write_json_atomic(progress_path, progress)
        if not path.exists() and not path.is_symlink():
            outcome = "already_absent_after_intent"
        else:
            _validate_file(path, expected)
            path.unlink()
            _fsync_directory(path.parent)
            outcome = "deleted"
        progress["file_status"][relative] = outcome
        progress["updated_at"] = _timestamp()
        _write_json_atomic(progress_path, progress)

    completed_directories = set(progress["completed_directories"])
    for relative in intent["directories"]:
        path = repository / relative
        if relative in completed_directories:
            if path.exists() or path.is_symlink():
                raise RuntimeError(
                    f"completed cleanup directory unexpectedly exists: {path}"
                )
            continue
        if path.is_symlink():
            raise RuntimeError(f"cleanup refuses directory symlink: {path}")
        if path.exists():
            try:
                path.rmdir()
            except OSError as exc:
                raise RuntimeError(
                    f"cleanup directory contains unexpected entries: {path}"
                ) from exc
            _fsync_directory(path.parent)
        completed_directories.add(relative)
        progress["completed_directories"] = sorted(completed_directories)
        progress["updated_at"] = _timestamp()
        _write_json_atomic(progress_path, progress)

    if any(path.exists() or path.is_symlink() for path in targets):
        raise RuntimeError("one or more superseded targets remain")
    audit_manifest = validate_publication_audit(
        audit,
        repository_root=repository,
        require_dependencies=False,
    )
    _require_bound_publication_unchanged(intent)
    if not all(path.exists() for path in protected if path != state):
        raise RuntimeError("a protected path is absent after cleanup")
    progress["status"] = "complete"
    progress["updated_at"] = _timestamp()
    _write_json_atomic(progress_path, progress)

    statuses = list(progress["file_status"].values())
    manifest: dict[str, Any] = {
        "schema_version": CLEANUP_MANIFEST_VERSION,
        "generated_at": _timestamp(),
        "intent_fingerprint": intent["intent_fingerprint"],
        "intent_sha256": _sha256(state / INTENT_FILENAME),
        "progress_sha256": _sha256(progress_path),
        "publication_audit_fingerprint": audit_manifest[
            "manifest_fingerprint"
        ],
        "publication_audit_manifest_sha256": _sha256(
            audit / MANIFEST_FILENAME
        ),
        "final_artifact_path": intent["final_artifact_path"],
        "final_artifact_sha256": intent["final_artifact_sha256"],
        "counts": {
            **intent["counts"],
            "deleted_files": statuses.count("deleted"),
            "already_absent_after_intent": statuses.count(
                "already_absent_after_intent"
            ),
        },
        "targets": intent["targets"],
        "validation": {
            "immutable_intent_valid": True,
            "publication_audit_valid_after_cleanup": True,
            "final_publication_unchanged": True,
            "all_allowlisted_targets_absent": True,
            "all_protected_paths_present": True,
            "all_files_terminal": all(
                status in {"deleted", "already_absent_after_intent"}
                for status in statuses
            ),
            "manifest_written_last": True,
        },
        "policy": {
            "only_exact_allowlisted_paths_removed": True,
            "no_recursive_wildcard_deletion": True,
            "unexpected_files_fail_closed": True,
            "resume_evidence_retained": True,
        },
    }
    if not all(manifest["validation"].values()):
        raise RuntimeError("superseded cleanup completion validation failed")
    manifest["manifest_fingerprint"] = canonical_semantic_fingerprint(
        manifest
    )
    _write_json_create_only(manifest_path, manifest)
    _fsync_directory(state)
    return manifest


def validate_superseded_cleanup(
    *,
    repository_root: str | Path,
    publication_audit_directory: str | Path,
    state_directory: str | Path,
) -> dict[str, Any]:
    """Validate a completed cleanup receipt and preserved publication."""

    repository = Path(repository_root).resolve()
    audit = Path(publication_audit_directory).resolve()
    state = Path(state_directory).resolve()
    expected_files = {
        state / INTENT_FILENAME,
        state / PROGRESS_FILENAME,
        state / MANIFEST_FILENAME,
    }
    observed_files = {
        path.resolve() for path in state.rglob("*") if path.is_file()
    }
    if observed_files != expected_files:
        raise RuntimeError("cleanup state file inventory is not exact")
    intent = _load_intent(state / INTENT_FILENAME)
    _require_invocation_matches(
        intent,
        repository=repository,
        audit=audit,
        state=state,
    )
    _validate_intent_contract(intent, repository)
    progress = _load_progress(state / PROGRESS_FILENAME, intent)
    if progress.get("status") != "complete":
        raise RuntimeError("cleanup progress is not complete")
    manifest_path = state / MANIFEST_FILENAME
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != CLEANUP_MANIFEST_VERSION:
        raise RuntimeError("cleanup manifest schema differs")
    fingerprint = manifest.get("manifest_fingerprint")
    body = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_fingerprint"
    }
    if fingerprint != canonical_semantic_fingerprint(body):
        raise RuntimeError("cleanup manifest fingerprint mismatch")
    if manifest.get("intent_fingerprint") != intent.get(
        "intent_fingerprint"
    ):
        raise RuntimeError("cleanup manifest intent binding differs")
    if manifest.get("intent_sha256") != _sha256(
        state / INTENT_FILENAME
    ):
        raise RuntimeError("cleanup intent checksum mismatch")
    if manifest.get("progress_sha256") != _sha256(
        state / PROGRESS_FILENAME
    ):
        raise RuntimeError("cleanup progress checksum mismatch")
    if manifest.get("publication_audit_manifest_sha256") != _sha256(
        audit / MANIFEST_FILENAME
    ):
        raise RuntimeError("cleanup publication audit checksum mismatch")
    if manifest.get("final_artifact_sha256") != intent.get(
        "final_artifact_sha256"
    ):
        raise RuntimeError("cleanup final-artifact binding differs")
    if manifest_path.stat().st_mtime_ns < max(
        (state / INTENT_FILENAME).stat().st_mtime_ns,
        (state / PROGRESS_FILENAME).stat().st_mtime_ns,
    ):
        raise RuntimeError("cleanup manifest was not written last")
    if not all(
        value is True for value in manifest["validation"].values()
    ):
        raise RuntimeError("cleanup manifest validation is not PASS")
    if any(
        status not in {"deleted", "already_absent_after_intent"}
        for status in progress["file_status"].values()
    ):
        raise RuntimeError("cleanup progress contains non-terminal files")
    if any(
        (repository / relative).exists()
        or (repository / relative).is_symlink()
        for relative in SUPERSEDED_RELATIVE_PATHS
    ):
        raise RuntimeError("a superseded cleanup target exists")
    protected = _protected_paths(
        repository,
        (
            audit,
            Path(intent["primary_manifest_path"]).parent,
            Path(intent["final_artifact_path"]),
            state,
        ),
    )
    if not all(path.exists() for path in protected):
        raise RuntimeError("a protected path is absent")
    audit_manifest = validate_publication_audit(
        audit,
        repository_root=repository,
        require_dependencies=False,
    )
    if audit_manifest["manifest_fingerprint"] != manifest.get(
        "publication_audit_fingerprint"
    ):
        raise RuntimeError("publication audit changed after cleanup")
    _require_bound_publication_unchanged(intent)
    return manifest


def _protected_paths(
    repository: Path,
    dynamic: Sequence[Path],
) -> tuple[Path, ...]:
    static = tuple(
        _exact_existing_path(repository, relative)
        for relative in PROTECTED_RELATIVE_PATHS
    )
    paths = tuple(path.resolve() for path in (*static, *dynamic))
    if any(not path.exists() for path in paths):
        missing = next(path for path in paths if not path.exists())
        raise FileNotFoundError(missing)
    return paths


def _validate_targets(
    targets: Sequence[Path],
    protected: Sequence[Path],
    repository: Path,
) -> None:
    _validate_target_contract(targets, protected, repository)
    for target in targets:
        if not target.is_dir() or target.is_symlink():
            raise RuntimeError(
                f"cleanup target is not a real directory: {target}"
            )


def _validate_target_contract(
    targets: Sequence[Path],
    protected: Sequence[Path],
    repository: Path,
) -> None:
    expected = {
        (repository / relative).resolve()
        for relative in SUPERSEDED_RELATIVE_PATHS
    }
    if set(targets) != expected or len(targets) != len(expected):
        raise RuntimeError("cleanup targets differ from exact allowlist")
    for target in targets:
        if not target.is_relative_to(repository) or target == repository:
            raise RuntimeError(
                f"cleanup target is outside repository: {target}"
            )
        for safe in protected:
            if (
                target == safe
                or target.is_relative_to(safe)
                or safe.is_relative_to(target)
            ):
                raise RuntimeError(
                    f"cleanup target intersects protected path: {target}"
                )
    for left_index, left in enumerate(targets):
        for right in targets[left_index + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise RuntimeError("cleanup allowlist targets overlap")


def _inventory_tree(
    repository: Path,
    target: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    files: list[dict[str, object]] = []
    directories: list[str] = []
    for root, directory_names, file_names in os.walk(
        target,
        topdown=True,
        followlinks=False,
    ):
        root_path = Path(root)
        root_stat = root_path.lstat()
        if stat.S_ISLNK(root_stat.st_mode):
            raise RuntimeError(f"cleanup refuses symlink: {root_path}")
        if not stat.S_ISDIR(root_stat.st_mode):
            raise RuntimeError(
                f"cleanup target contains non-directory root: {root_path}"
            )
        directories.append(str(root_path.relative_to(repository)))
        for name in sorted(directory_names):
            child = root_path / name
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                raise RuntimeError(f"cleanup refuses symlink: {child}")
            if not stat.S_ISDIR(child_stat.st_mode):
                raise RuntimeError(
                    f"cleanup target contains unsupported object: {child}"
                )
        for name in sorted(file_names):
            child = root_path / name
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                raise RuntimeError(f"cleanup refuses symlink: {child}")
            if not stat.S_ISREG(child_stat.st_mode):
                raise RuntimeError(
                    f"cleanup target contains unsupported object: {child}"
                )
            files.append(
                {
                    "relative_path": str(child.relative_to(repository)),
                    "physical_bytes": child_stat.st_size,
                    "physical_sha256": _sha256(child),
                }
            )
    files.sort(key=lambda item: str(item["relative_path"]))
    return files, directories


def _validate_file(
    path: Path,
    expected: Mapping[str, object],
) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"cleanup file type changed: {path}")
    if info.st_size != int(expected["physical_bytes"]):
        raise RuntimeError(f"cleanup file size changed: {path}")
    if _sha256(path) != expected["physical_sha256"]:
        raise RuntimeError(f"cleanup file checksum changed: {path}")


def _load_intent(path: Path) -> dict[str, Any]:
    intent = _load_json(path)
    if intent.get("schema_version") != CLEANUP_INTENT_VERSION:
        raise RuntimeError("cleanup intent schema differs")
    fingerprint = intent.get("intent_fingerprint")
    body = {
        key: value
        for key, value in intent.items()
        if key != "intent_fingerprint"
    }
    if fingerprint != canonical_semantic_fingerprint(body):
        raise RuntimeError("cleanup intent fingerprint mismatch")
    if not all(value is True for value in intent["validation"].values()):
        raise RuntimeError("cleanup intent validation is not PASS")
    return intent


def _load_progress(
    path: Path,
    intent: Mapping[str, object],
) -> dict[str, Any]:
    progress = _load_json(path)
    if progress.get("schema_version") != CLEANUP_PROGRESS_VERSION:
        raise RuntimeError("cleanup progress schema differs")
    if progress.get("intent_fingerprint") != intent.get(
        "intent_fingerprint"
    ):
        raise RuntimeError("cleanup progress intent binding differs")
    expected_files = {
        str(item["relative_path"]) for item in intent["files"]
    }
    statuses = progress.get("file_status")
    if not isinstance(statuses, dict) or set(statuses) != expected_files:
        raise RuntimeError("cleanup progress file inventory differs")
    allowed = {
        "pending",
        "deleting",
        "deleted",
        "already_absent_after_intent",
    }
    if any(value not in allowed for value in statuses.values()):
        raise RuntimeError("cleanup progress contains invalid status")
    if progress.get("status") not in {"prepared", "running", "complete"}:
        raise RuntimeError("cleanup progress run status is invalid")
    completed_directories = progress.get("completed_directories")
    if not isinstance(completed_directories, list) or len(
        completed_directories
    ) != len(set(completed_directories)):
        raise RuntimeError("cleanup progress directory evidence is invalid")
    if not set(completed_directories).issubset(intent["directories"]):
        raise RuntimeError("cleanup progress has an unknown directory")
    return progress


def _validate_intent_contract(
    intent: Mapping[str, object],
    repository: Path,
) -> None:
    target_rows = intent.get("targets")
    if not isinstance(target_rows, list):
        raise RuntimeError("cleanup intent target evidence is invalid")
    target_names = [
        str(item.get("relative_path"))
        for item in target_rows
        if isinstance(item, dict)
    ]
    if target_names != list(SUPERSEDED_RELATIVE_PATHS):
        raise RuntimeError("cleanup intent targets differ from allowlist")
    target_paths = [Path(relative) for relative in target_names]

    files = intent.get("files")
    if not isinstance(files, list):
        raise RuntimeError("cleanup intent file evidence is invalid")
    file_names: list[str] = []
    grouped: dict[str, list[dict[str, object]]] = {
        relative: [] for relative in target_names
    }
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError("cleanup intent file evidence is invalid")
        relative = str(item.get("relative_path") or "")
        path = Path(relative)
        if (
            not relative
            or path.is_absolute()
            or ".." in path.parts
            or not any(
                path.is_relative_to(target)
                and path != target
                for target in target_paths
            )
        ):
            raise RuntimeError(
                f"cleanup intent file is outside allowlist: {relative}"
            )
        try:
            physical_bytes = int(item["physical_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "cleanup intent file size is invalid"
            ) from exc
        checksum = str(item.get("physical_sha256") or "")
        if physical_bytes < 0 or not checksum.startswith("sha256:"):
            raise RuntimeError(
                "cleanup intent file inventory is invalid"
            )
        file_names.append(relative)
        target_name = next(
            name
            for name, target in zip(
                target_names,
                target_paths,
                strict=True,
            )
            if path.is_relative_to(target) and path != target
        )
        grouped[target_name].append(dict(item))
    if file_names != sorted(file_names) or len(file_names) != len(
        set(file_names)
    ):
        raise RuntimeError(
            "cleanup intent files are not unique and ordered"
        )

    directories = intent.get("directories")
    if not isinstance(directories, list) or len(directories) != len(
        set(directories)
    ):
        raise RuntimeError("cleanup intent directory evidence is invalid")
    directory_paths = [Path(str(value)) for value in directories]
    if not all(
        not path.is_absolute()
        and ".." not in path.parts
        and any(path.is_relative_to(target) for target in target_paths)
        for path in directory_paths
    ):
        raise RuntimeError(
            "cleanup intent directory is outside allowlist"
        )
    if not set(target_names).issubset(
        str(path) for path in directory_paths
    ):
        raise RuntimeError("cleanup intent omits a target directory")

    for target in target_rows:
        name = str(target["relative_path"])
        evidence = sorted(
            grouped[name],
            key=lambda item: str(item["relative_path"]),
        )
        if int(target.get("file_count", -1)) != len(evidence):
            raise RuntimeError("cleanup intent target file count differs")
        if int(target.get("physical_bytes", -1)) != sum(
            int(item["physical_bytes"]) for item in evidence
        ):
            raise RuntimeError("cleanup intent target byte count differs")
        if target.get(
            "tree_fingerprint"
        ) != canonical_semantic_fingerprint(evidence):
            raise RuntimeError(
                "cleanup intent target fingerprint differs"
            )
    counts = intent.get("counts")
    if not isinstance(counts, dict) or counts != {
        "target_directories": len(target_rows),
        "files": len(files),
        "directories": len(directories),
        "physical_bytes": sum(
            int(item["physical_bytes"]) for item in files
        ),
    }:
        raise RuntimeError("cleanup intent aggregate counts differ")
    if Path(str(intent.get("repository_root"))).resolve() != repository:
        raise RuntimeError("cleanup intent repository binding differs")


def _require_invocation_matches(
    intent: Mapping[str, object],
    *,
    repository: Path,
    audit: Path,
    state: Path,
) -> None:
    if Path(str(intent.get("repository_root"))).resolve() != repository:
        raise RuntimeError("cleanup repository differs from intent")
    if (
        Path(str(intent.get("publication_audit_directory"))).resolve()
        != audit
    ):
        raise RuntimeError("cleanup publication audit differs from intent")
    if Path(str(intent.get("state_directory"))).resolve() != state:
        raise RuntimeError("cleanup state directory differs from intent")


def _require_bound_publication_unchanged(
    intent: Mapping[str, object],
) -> None:
    primary_manifest = Path(str(intent["primary_manifest_path"]))
    final_artifact = Path(str(intent["final_artifact_path"]))
    if _sha256(primary_manifest) != intent["primary_manifest_sha256"]:
        raise RuntimeError("primary manifest changed during cleanup")
    if _sha256(final_artifact) != intent["final_artifact_sha256"]:
        raise RuntimeError("final artifact changed during cleanup")


def _exact_existing_path(repository: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"unsafe cleanup path: {relative}")
    lexical = repository / relative_path
    if not lexical.exists() and not lexical.is_symlink():
        raise FileNotFoundError(lexical)
    resolved = lexical.resolve(strict=True)
    if resolved != lexical.absolute():
        raise RuntimeError(f"cleanup path traverses a symlink: {lexical}")
    return resolved


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir() or path.is_symlink():
        raise FileNotFoundError(f"{label}: {path}")


def _require_no_overlap(
    path: Path,
    others: Sequence[Path],
    *,
    context: str,
    allow_ancestor: Path,
) -> None:
    if not path.is_relative_to(allow_ancestor):
        raise RuntimeError(f"{context} must be inside repository")
    for other in others:
        if other == allow_ancestor:
            continue
        if (
            path == other
            or path.is_relative_to(other)
            or other.is_relative_to(path)
        ):
            raise RuntimeError(f"{context} overlaps protected scope")


def _display_path(repository: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repository))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read cleanup evidence: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"cleanup evidence is not an object: {path}")
    return value


def _write_json_create_only(
    path: Path,
    value: Mapping[str, object],
) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode()
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _write_json_atomic(
    path: Path,
    value: Mapping[str, object],
) -> None:
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        _write_json_create_only(temporary, value)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_empty_staging(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_file() and not child.is_symlink():
            child.unlink()
    path.rmdir()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CLEANUP_INTENT_VERSION",
    "CLEANUP_MANIFEST_VERSION",
    "CLEANUP_PROGRESS_VERSION",
    "PROTECTED_RELATIVE_PATHS",
    "SUPERSEDED_RELATIVE_PATHS",
    "execute_superseded_cleanup",
    "plan_superseded_cleanup",
    "prepare_superseded_cleanup",
    "validate_superseded_cleanup",
]
