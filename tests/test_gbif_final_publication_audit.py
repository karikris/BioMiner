from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_final.pipeline import (
    FINAL_FILENAME,
    FINAL_SCHEMA_VERSION,
    MANIFEST_FILENAME,
)
from biominer.gbif_final.locator_index import (
    DATABASE_FILENAME,
    build_final_locator_index,
    validate_final_locator_index,
)
from biominer.gbif_final.publication_audit import (
    PUBLICATION_AUDIT_VERSION,
    audit_final_publication,
    validate_publication_audit,
)


def _write(
    path: Path,
    values: dict[str, list[object]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(values),
        path,
        compression="zstd",
        row_group_size=1,
    )
    return path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, object]:
    repository = Path(__file__).parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    temporal = _write(
        tmp_path / "inputs" / "temporal.parquet",
        {"gbifID": ["A", "B"], "species": ["Alpha", "Beta"]},
    )
    pre_temporal = _write(
        tmp_path / "inputs" / "pre-temporal.parquet",
        {"gbifID": ["A", "X", "B"]},
    )
    registry = tmp_path / "registry"
    _write(registry / "taxa.parquet", {"key": ["1"]})
    _write(registry / "names.parquet", {"name": ["Alpha"]})
    _write(registry / "species_paths.parquet", {"key": ["1"]})
    assertions = _write(
        tmp_path / "inputs" / "source-assertions.parquet",
        {"assertion_id": ["a1"]},
    )
    quality = tmp_path / "quality"
    quality_paths = (
        quality
        / "media_assertion_quality"
        / "media_assertion_quality.parquet",
        quality / "occurrence_quality" / "occurrence_quality.parquet",
        quality / "rights_and_attribution" / "media_rights.parquet",
        quality / "duplicates" / "duplicate_membership.parquet",
        quality
        / "quality_results"
        / "phase3"
        / "derived_assertions.parquet",
        quality / "ai_readiness" / "parts" / "part-00000.parquet",
    )
    for index, path in enumerate(quality_paths):
        _write(path, {"value": [index]})

    publication = tmp_path / "publication"
    publication.mkdir()
    final_path = _write(
        publication / FINAL_FILENAME,
        {
            "gbifID": ["A", "B"],
            "source_row_id": ["source-1", "source-2"],
            "media_assertion_id": ["media-1", "media-2"],
            "media_identifier": [
                "https://example.test/a.jpg",
                "https://example.test/b.jpg",
            ],
            "media_references": [
                "https://example.test/a",
                "https://example.test/b",
            ],
            "occurrence_quality": ["pass", "pass"],
            "media_quality": ["pass", "pass"],
            "rights_quality": ["pass", "pass"],
            "duplicate_quality": ["unique", "unique"],
            "ai_readiness": ["review", "review"],
            "derived_quality_assertions": [None, None],
            "registry_match_status": ["matched", "matched"],
            "registry_match_method": ["key", "key"],
            "registry_taxon_key": ["taxon-1", "taxon-2"],
            "speciesKey": ["1", "2"],
            "species": ["Alpha", "Beta"],
            "keyword_evidence": [["alpha"], ["beta"]],
            "keyword_source_assertions": [[], []],
            "flickr_query_terms": [["alpha"], ["beta"]],
        },
    )
    metadata = pq.ParquetFile(final_path).metadata
    manifest = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "producer_git_sha": commit,
        "artifact": {
            "path": FINAL_FILENAME,
            "rows": metadata.num_rows,
            "columns": metadata.num_columns,
            "row_groups": metadata.num_row_groups,
            "row_group_rows": [
                metadata.row_group(index).num_rows
                for index in range(metadata.num_row_groups)
            ],
            "bytes": final_path.stat().st_size,
            "sha256": _sha256(final_path),
        },
        "inputs": {
            "temporal_parquet": {
                "path": str(temporal),
                "sha256": _sha256(temporal),
            },
            "pre_temporal_parquet": {
                "path": str(pre_temporal),
                "sha256": _sha256(pre_temporal),
            },
            "registry_dir": str(registry),
            "quality_dir": str(quality),
        },
        "acceptance_gate": {
            "row_count_preserved": True,
            "stable_media_identity_complete": True,
            "row_groups_complete": True,
            "manifest_written_last": True,
        },
    }
    (publication / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "publication_directory": publication,
        "temporal_parquet": temporal,
        "pre_temporal_parquet": pre_temporal,
        "registry_directory": registry,
        "source_assertions": assertions,
        "quality_directory": quality,
        "output_directory": tmp_path / "audit",
        "repository_root": repository,
        "expected_producer_git_sha": commit,
        "memory_limit": "1GB",
        "threads": 1,
    }


def test_publication_audit_binds_legacy_dependencies_and_identities(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    manifest = audit_final_publication(**arguments)
    audit = Path(str(arguments["output_directory"]))

    assert manifest["schema_version"] == PUBLICATION_AUDIT_VERSION
    assert manifest["audit_git_commit"] == arguments[
        "expected_producer_git_sha"
    ]
    assert manifest["counts"]["rows"] == 2
    assert manifest["identity_audit"]["unique_source_row_ids"] == 2
    assert all(manifest["validation"].values())
    assert set(path.name for path in audit.iterdir()) == {
        "input_inventory.parquet",
        "identity_audit.parquet",
        "manifest.json",
    }
    disk = json.loads((audit / "manifest.json").read_text())
    assert disk == manifest
    assert (
        validate_publication_audit(
            audit,
            repository_root=arguments["repository_root"],
        )
        == manifest
    )
    assert {
        artifact["path"]
        for artifact in manifest["artifacts"].values()
    } == {
        "input_inventory.parquet",
        "identity_audit.parquet",
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit_final_publication(**arguments)


def test_publication_audit_rejects_changed_input_after_build(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    temporal = Path(str(arguments["temporal_parquet"]))
    pq.write_table(
        pa.table(
            {"gbifID": ["A", "B"], "species": ["Changed", "Beta"]}
        ),
        temporal,
        compression="zstd",
    )

    with pytest.raises(
        RuntimeError,
        match="final publication audit failed",
    ):
        audit_final_publication(**arguments)


def test_publication_audit_validator_rejects_changed_dependency(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    audit_final_publication(**arguments)
    temporal = Path(str(arguments["temporal_parquet"]))
    pq.write_table(
        pa.table(
            {"gbifID": ["A", "B"], "species": ["Changed", "Beta"]}
        ),
        temporal,
        compression="zstd",
    )

    with pytest.raises(
        RuntimeError,
        match="audited dependency temporal inventory mismatch",
    ):
        validate_publication_audit(
            arguments["output_directory"],
            repository_root=arguments["repository_root"],
        )
    validate_publication_audit(
        arguments["output_directory"],
        repository_root=arguments["repository_root"],
        require_dependencies=False,
    )


def test_publication_audit_validator_rejects_changed_audit_artifact(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    audit_final_publication(**arguments)
    identity_path = (
        Path(str(arguments["output_directory"]))
        / "identity_audit.parquet"
    )
    identity_path.write_bytes(identity_path.read_bytes() + b"tampered")

    with pytest.raises(
        RuntimeError,
        match="cannot inspect Parquet artifact",
    ):
        validate_publication_audit(
            arguments["output_directory"],
            repository_root=arguments["repository_root"],
        )


def test_publication_audit_validator_accepts_relocated_publication(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    manifest = audit_final_publication(**arguments)
    original = Path(str(arguments["publication_directory"]))
    relocated = tmp_path / "received" / "current"
    shutil.copytree(original, relocated)
    shutil.rmtree(original)

    with pytest.raises(
        FileNotFoundError,
        match="audited final publication is no longer complete",
    ):
        validate_publication_audit(
            arguments["output_directory"],
            repository_root=arguments["repository_root"],
            require_dependencies=False,
        )
    assert (
        validate_publication_audit(
            arguments["output_directory"],
            repository_root=arguments["repository_root"],
            require_dependencies=False,
            primary_publication_directory=relocated,
        )
        == manifest
    )


def test_final_locator_indexes_validated_publication_without_full_copy(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    audit_final_publication(**arguments)
    locator = tmp_path / "locator"

    manifest = build_final_locator_index(
        publication_directory=arguments["publication_directory"],
        publication_audit_directory=arguments["output_directory"],
        output_directory=locator,
        repository_root=arguments["repository_root"],
        memory_limit="1GB",
        threads=1,
    )

    assert all(manifest["validation"].values())
    assert manifest["policy"][
        "full_enriched_rows_remain_only_in_parquet"
    ]
    assert manifest["database"]["row_count"] == 2
    assert set(path.name for path in locator.iterdir()) == {
        DATABASE_FILENAME,
        "manifest.json",
    }
    assert (
        validate_final_locator_index(
            index_directory=locator,
            publication_audit_directory=arguments["output_directory"],
            publication_directory=arguments["publication_directory"],
            repository_root=arguments["repository_root"],
            require_dependencies=True,
        )
        == manifest
    )
    connection = duckdb.connect(
        str(locator / DATABASE_FILENAME),
        read_only=True,
    )
    try:
        assert connection.execute(
            """
            SELECT media_identifier
            FROM media_locator
            WHERE speciesKey = ?
            """,
            ["1"],
        ).fetchall() == [("https://example.test/a.jpg",)]
    finally:
        connection.close()


def test_final_locator_validator_rejects_changed_database(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    audit_final_publication(**arguments)
    locator = tmp_path / "locator"
    build_final_locator_index(
        publication_directory=arguments["publication_directory"],
        publication_audit_directory=arguments["output_directory"],
        output_directory=locator,
        repository_root=arguments["repository_root"],
        memory_limit="1GB",
        threads=1,
    )
    database = locator / DATABASE_FILENAME
    database.write_bytes(database.read_bytes() + b"tampered")

    with pytest.raises(
        RuntimeError,
        match="locator database checksum mismatch",
    ):
        validate_final_locator_index(
            index_directory=locator,
            publication_audit_directory=arguments["output_directory"],
            publication_directory=arguments["publication_directory"],
            repository_root=arguments["repository_root"],
            require_dependencies=True,
        )
