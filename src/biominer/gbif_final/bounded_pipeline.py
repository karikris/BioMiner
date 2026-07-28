from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

import duckdb
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.gbif_final.bounded import (
    assemble_parts,
    validate_assembled_output,
)
from biominer.gbif_final.materialize import (
    seal_temporal_enriched_window,
)
from biominer.gbif_final.spine import validate_source_spine
from biominer.gbif_final.windowed import (
    seal_keyed_dimension_window,
    seal_null_safe_composite_dimension_window,
    seal_ordinal_aligned_window,
)


BOUNDED_PIPELINE_VERSION = "gbif-final-bounded-pipeline/v1"
CHECKPOINT_VERSION = "gbif-final-bounded-pipeline-checkpoint/v1"
CHECKPOINT_FILENAME = "checkpoint.json"

SPECIES_OUTPUT_FIELDS = (
    "registry_match_status",
    "registry_match_method",
    "registry_taxon_key",
    "keyword_evidence",
    "keyword_source_assertions",
    "flickr_query_terms",
)


def build_bounded_final_from_spine(
    *,
    temporal_parquet: str | Path,
    source_spine_directory: str | Path,
    media_quality_parquet: str | Path,
    occurrence_quality_parquet: str | Path,
    rights_quality_parquet: str | Path,
    duplicate_quality_parquet: str | Path,
    ai_readiness_parts: Sequence[str | Path],
    derived_assertion_dimension: str | Path,
    species_enrichment_dimension: str | Path,
    work_directory: str | Path,
    output_directory: str | Path,
    producer_git_sha: str,
    threads: int = 4,
    memory_limit: str = "8GB",
    batch_rows: int = 65_536,
    final_row_group_size: int = 100_000,
    free_space_multiplier: float = 1.25,
    minimum_headroom_bytes: int = 2 * 1024**3,
) -> dict[str, Any]:
    """Build and publish the final enriched dataset in restartable windows."""

    if not producer_git_sha.strip():
        raise ValueError("producer_git_sha must be non-empty")
    if threads <= 0:
        raise ValueError("threads must be positive")
    if batch_rows <= 0 or final_row_group_size <= 0:
        raise ValueError("row batch sizes must be positive")
    if not memory_limit.strip():
        raise ValueError("memory_limit must be non-empty")
    if not ai_readiness_parts:
        raise ValueError("ai_readiness_parts must be non-empty")

    temporal_path = Path(temporal_parquet).resolve()
    spine_directory = Path(source_spine_directory).resolve()
    work = Path(work_directory).resolve()
    output = Path(output_directory).resolve()
    if (
        work == output
        or work.is_relative_to(output)
        or output.is_relative_to(work)
    ):
        raise ValueError("work and output directories must not overlap")
    if output.exists() and not output.is_dir():
        raise FileExistsError(output)

    spine_manifest = validate_source_spine(spine_directory)
    expected_rows = int(spine_manifest["counts"]["post_1960_rows"])
    source_scope = {
        "publication_role": (
            "user_authorized_legacy_consolidation_source_of_truth"
        ),
        "ground_zero_production_lineage": False,
        "baseline": "legacy_v3_rights_filtered",
        "baseline_rows": int(
            spine_manifest["counts"]["pre_temporal_rows"]
        ),
        "row_scope": "post_1960",
        "excluded_pre_1960_rows": int(
            spine_manifest["counts"]["excluded_pre_1960_rows"]
        ),
        "rows": expected_rows,
        "source_spine_run_fingerprint": spine_manifest["run_fingerprint"],
    }

    dimension_paths: dict[str, Path | tuple[Path, ...]] = {
        "temporal": temporal_path,
        "media_quality": Path(media_quality_parquet).resolve(),
        "occurrence_quality": Path(occurrence_quality_parquet).resolve(),
        "rights_quality": Path(rights_quality_parquet).resolve(),
        "duplicate_quality": Path(duplicate_quality_parquet).resolve(),
        "ai_readiness": tuple(
            sorted(Path(path).resolve() for path in ai_readiness_parts)
        ),
        "derived_assertions": Path(
            derived_assertion_dimension
        ).resolve(),
        "species_enrichment": Path(
            species_enrichment_dimension
        ).resolve(),
    }
    inventories = {
        name: (
            [_parquet_inventory(path) for path in paths]
            if isinstance(paths, tuple)
            else _parquet_inventory(paths)
        )
        for name, paths in dimension_paths.items()
    }
    if int(inventories["temporal"]["row_count"]) != expected_rows:
        raise RuntimeError(
            "temporal and source-spine row counts differ: "
            f"temporal={inventories['temporal']['row_count']}, "
            f"spine={expected_rows}"
        )
    for name in ("temporal", "media_quality"):
        if inventories[name] != spine_manifest["input_inventory"][name]:
            raise RuntimeError(
                f"{name} does not match the source-spine input inventory"
            )
    semantic_config = {
        "schema_version": BOUNDED_PIPELINE_VERSION,
        "producer_git_sha": producer_git_sha,
        "source_spine_manifest_fingerprint": spine_manifest[
            "manifest_fingerprint"
        ],
        "source_scope": dict(source_scope),
        "inputs": inventories,
        "batch_rows": batch_rows,
        "final_row_group_size": final_row_group_size,
    }
    run_fingerprint = canonical_semantic_fingerprint(semantic_config)
    source_scope["bounded_pipeline_run_fingerprint"] = run_fingerprint
    common_dependencies = {
        "bounded_pipeline_version": BOUNDED_PIPELINE_VERSION,
        "run_fingerprint": run_fingerprint,
        "producer_git_sha": producer_git_sha,
        "source_spine_manifest_fingerprint": spine_manifest[
            "manifest_fingerprint"
        ],
    }

    _prepare_work_directory(
        work=work,
        run_fingerprint=run_fingerprint,
        semantic_config=semantic_config,
    )
    if output.exists():
        return validate_assembled_output(
            output,
            expected_rows=expected_rows,
            expected_code_commit=producer_git_sha,
            expected_source_scope=source_scope,
        )

    temporary = work / ".duckdb_tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir()
    connection = duckdb.connect()
    final_receipts: list[Path] = []
    expected_work_files = {work / CHECKPOINT_FILENAME}
    try:
        connection.execute(f"SET threads={int(threads)}")
        connection.execute("SET memory_limit=?", [memory_limit])
        connection.execute("SET temp_directory=?", [str(temporary)])
        connection.execute("SET preserve_insertion_order=false")

        for part_index, evidence in enumerate(
            spine_manifest["part_evidence"]
        ):
            start = int(evidence["source_start_ordinal"])
            stop = int(evidence["source_stop_ordinal"])
            spine_part = (
                spine_directory / str(evidence["part_path"])
            ).resolve()
            part_directory = (
                work / "windows" / f"part-{part_index:05d}"
            )
            part_directory.mkdir(parents=True, exist_ok=True)
            spine_dependencies = {
                **common_dependencies,
                "source_spine_part_id": evidence["part_id"],
                "source_spine_part_sha256": evidence["part_sha256"],
            }
            sidecars: dict[str, Path] = {}
            sidecar_receipts: dict[str, dict[str, Any]] = {}

            sidecars["occurrence_quality"] = _seal_keyed_sidecar(
                connection=connection,
                spine_part=spine_part,
                dimension=dimension_paths["occurrence_quality"],
                output_part=part_directory
                / "occurrence_quality.parquet",
                start=start,
                stop=stop,
                spine_key="gbifID",
                dimension_key="gbifID",
                output_column="occurrence_quality",
                excluded={"gbifID"},
                required_match=True,
                dependencies={
                    **spine_dependencies,
                    "dimension_inventory": inventories[
                        "occurrence_quality"
                    ],
                },
                batch_rows=batch_rows,
                expected_work_files=expected_work_files,
                receipt_registry=sidecar_receipts,
            )
            sidecars["media_quality"] = _seal_keyed_sidecar(
                connection=connection,
                spine_part=spine_part,
                dimension=dimension_paths["media_quality"],
                output_part=part_directory / "media_quality.parquet",
                start=start,
                stop=stop,
                spine_key="media_assertion_id",
                dimension_key="media_assertion_id",
                output_column="media_quality",
                excluded={
                    "source_row_id",
                    "media_assertion_id",
                    "gbifID",
                    "source_sort_position",
                },
                required_match=True,
                dependencies={
                    **spine_dependencies,
                    "dimension_inventory": inventories["media_quality"],
                },
                batch_rows=batch_rows,
                expected_work_files=expected_work_files,
                receipt_registry=sidecar_receipts,
            )
            sidecars["rights_quality"] = _seal_keyed_sidecar(
                connection=connection,
                spine_part=spine_part,
                dimension=dimension_paths["rights_quality"],
                output_part=part_directory / "rights_quality.parquet",
                start=start,
                stop=stop,
                spine_key="media_assertion_id",
                dimension_key="media_assertion_id",
                output_column="rights_quality",
                excluded={
                    "source_row_id",
                    "media_assertion_id",
                    "gbifID",
                    "media_identifier",
                },
                required_match=True,
                dependencies={
                    **spine_dependencies,
                    "dimension_inventory": inventories["rights_quality"],
                },
                batch_rows=batch_rows,
                expected_work_files=expected_work_files,
                receipt_registry=sidecar_receipts,
            )
            sidecars["duplicate_quality"] = _seal_keyed_sidecar(
                connection=connection,
                spine_part=spine_part,
                dimension=dimension_paths["duplicate_quality"],
                output_part=part_directory / "duplicate_quality.parquet",
                start=start,
                stop=stop,
                spine_key="media_assertion_id",
                dimension_key="media_assertion_id",
                output_column="duplicate_quality",
                excluded={
                    "source_row_id",
                    "media_assertion_id",
                    "gbifID",
                },
                required_match=True,
                dependencies={
                    **spine_dependencies,
                    "dimension_inventory": inventories[
                        "duplicate_quality"
                    ],
                },
                batch_rows=batch_rows,
                expected_work_files=expected_work_files,
                receipt_registry=sidecar_receipts,
            )
            sidecars["ai_readiness"] = _seal_keyed_sidecar(
                connection=connection,
                spine_part=spine_part,
                dimension=dimension_paths["ai_readiness"],
                output_part=part_directory / "ai_readiness.parquet",
                start=start,
                stop=stop,
                spine_key="media_assertion_id",
                dimension_key="media_assertion_id",
                output_column="ai_readiness",
                excluded={
                    "source_row_id",
                    "media_assertion_id",
                    "gbifID",
                },
                required_match=True,
                dependencies={
                    **spine_dependencies,
                    "dimension_inventory": inventories["ai_readiness"],
                },
                batch_rows=batch_rows,
                expected_work_files=expected_work_files,
                receipt_registry=sidecar_receipts,
            )
            sidecars["derived_quality"] = _seal_keyed_sidecar(
                connection=connection,
                spine_part=spine_part,
                dimension=dimension_paths["derived_assertions"],
                output_part=part_directory / "derived_quality.parquet",
                start=start,
                stop=stop,
                spine_key="gbifID",
                dimension_key="gbifID",
                output_column="derived_quality",
                excluded={"dimension_ordinal", "gbifID"},
                required_match=False,
                dependencies={
                    **spine_dependencies,
                    "dimension_inventory": inventories[
                        "derived_assertions"
                    ],
                },
                batch_rows=batch_rows,
                expected_work_files=expected_work_files,
                receipt_registry=sidecar_receipts,
            )

            species_path = part_directory / "species_enrichment.parquet"
            species_receipt = seal_null_safe_composite_dimension_window(
                connection=connection,
                spine_part=spine_part,
                dimension=dimension_paths["species_enrichment"],
                output_part=species_path,
                source_start_ordinal=start,
                source_stop_ordinal=stop,
                key_pairs=(
                    ("speciesKey", "dataset_species_key"),
                    ("species", "dataset_species"),
                ),
                output_column="species_enrichment",
                excluded_dimension_columns={
                    "dataset_species_key",
                    "dataset_species",
                },
                required_match=True,
                dependencies={
                    **spine_dependencies,
                    "dimension_inventory": inventories[
                        "species_enrichment"
                    ],
                },
                batch_rows=batch_rows,
            )
            sidecars["species_enrichment"] = species_path
            sidecar_receipts["species_enrichment"] = species_receipt
            _record_sealed_files(
                expected_work_files,
                species_path,
                species_receipt,
            )

            aligned_path = part_directory / "aligned_enrichments.parquet"
            aligned_receipt = seal_ordinal_aligned_window(
                connection=connection,
                spine_part=spine_part,
                sidecar_parts=sidecars,
                output_part=aligned_path,
                source_start_ordinal=start,
                source_stop_ordinal=stop,
                spine_columns=(
                    "source_row_id",
                    "media_assertion_id",
                ),
                dependencies={
                    **spine_dependencies,
                    "sidecar_part_ids": {
                        name: sidecar_receipts[name]["part_id"]
                        for name in sidecars
                    },
                },
                batch_rows=batch_rows,
            )
            _record_sealed_files(
                expected_work_files,
                aligned_path,
                aligned_receipt,
            )

            final_part = part_directory / "final.parquet"
            final_receipt = seal_temporal_enriched_window(
                connection=connection,
                temporal_parquet=temporal_path,
                aligned_part=aligned_path,
                output_part=final_part,
                source_start_ordinal=start,
                source_stop_ordinal=stop,
                expanded_struct_fields={
                    "derived_quality": (
                        "derived_quality_assertions",
                    ),
                    "species_enrichment": SPECIES_OUTPUT_FIELDS,
                },
                dependencies={
                    **spine_dependencies,
                    "temporal_inventory": inventories["temporal"],
                    "aligned_part_id": aligned_receipt["part_id"],
                    "aligned_part_sha256": aligned_receipt["artifact"][
                        "physical_sha256"
                    ],
                },
                batch_rows=batch_rows,
            )
            _record_sealed_files(
                expected_work_files,
                final_part,
                final_receipt,
            )
            final_receipts.append(
                final_part.with_suffix(".parquet.receipt.json")
            )
    finally:
        connection.close()
        shutil.rmtree(temporary, ignore_errors=True)

    _reject_unexpected_work_files(
        work=work,
        expected=expected_work_files,
    )
    manifest = assemble_parts(
        part_receipts=final_receipts,
        output_directory=output,
        expected_rows=expected_rows,
        code_commit=producer_git_sha,
        source_scope=source_scope,
        row_group_size=final_row_group_size,
        free_space_multiplier=free_space_multiplier,
        minimum_headroom_bytes=minimum_headroom_bytes,
    )
    validated = validate_assembled_output(
        output,
        expected_rows=expected_rows,
        expected_code_commit=producer_git_sha,
        expected_source_scope=source_scope,
    )
    if validated != manifest:
        raise RuntimeError("bounded final validation changed the manifest")
    return manifest


def _seal_keyed_sidecar(
    *,
    connection: duckdb.DuckDBPyConnection,
    spine_part: Path,
    dimension: Path | tuple[Path, ...],
    output_part: Path,
    start: int,
    stop: int,
    spine_key: str,
    dimension_key: str,
    output_column: str,
    excluded: set[str],
    required_match: bool,
    dependencies: Mapping[str, object],
    batch_rows: int,
    expected_work_files: set[Path],
    receipt_registry: dict[str, dict[str, Any]],
) -> Path:
    receipt = seal_keyed_dimension_window(
        connection=connection,
        spine_part=spine_part,
        dimension=dimension,
        output_part=output_part,
        source_start_ordinal=start,
        source_stop_ordinal=stop,
        spine_key=spine_key,
        dimension_key=dimension_key,
        output_column=output_column,
        excluded_dimension_columns=excluded,
        required_match=required_match,
        dependencies=dependencies,
        batch_rows=batch_rows,
    )
    _record_sealed_files(
        expected_work_files,
        output_part,
        receipt,
    )
    receipt_registry[output_column] = receipt
    return output_part


def _record_sealed_files(
    expected: set[Path],
    part: Path,
    receipt: Mapping[str, object],
) -> None:
    if (
        not part.is_file()
        or receipt["artifact"]["path"] != part.name
        or not part.with_suffix(
            part.suffix + ".receipt.json"
        ).is_file()
    ):
        raise RuntimeError(f"sealed part receipt is incomplete: {part}")
    expected.add(part.resolve())
    expected.add(
        part.with_suffix(part.suffix + ".receipt.json").resolve()
    )


def _prepare_work_directory(
    *,
    work: Path,
    run_fingerprint: str,
    semantic_config: Mapping[str, object],
) -> None:
    checkpoint_path = work / CHECKPOINT_FILENAME
    if work.exists():
        if not work.is_dir() or not checkpoint_path.is_file():
            raise RuntimeError(
                f"bounded work directory lacks checkpoint: {work}"
            )
        checkpoint = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )
        if (
            checkpoint.get("schema_version") != CHECKPOINT_VERSION
            or checkpoint.get("run_fingerprint") != run_fingerprint
            or checkpoint.get("semantic_config") != semantic_config
            or checkpoint.get("checkpoint_fingerprint")
            != _checkpoint_fingerprint(checkpoint)
        ):
            raise RuntimeError(
                f"bounded work checkpoint is stale: {checkpoint_path}"
            )
        return

    work.mkdir(parents=True)
    checkpoint: dict[str, object] = {
        "schema_version": CHECKPOINT_VERSION,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "run_fingerprint": run_fingerprint,
        "semantic_config": semantic_config,
        "checkpoint_policy": {
            "create_only": True,
            "sealed_parts_are_restart_authority": True,
        },
    }
    checkpoint["checkpoint_fingerprint"] = _checkpoint_fingerprint(
        checkpoint
    )
    _write_json_create_only(checkpoint_path, checkpoint)


def _reject_unexpected_work_files(
    *,
    work: Path,
    expected: set[Path],
) -> None:
    observed = {
        path.resolve()
        for path in work.rglob("*")
        if path.is_file()
    }
    if observed != {path.resolve() for path in expected}:
        raise RuntimeError(
            "bounded work file inventory differs from sealed receipts"
        )


def _parquet_inventory(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    row_group_rows = [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ]
    if (
        parquet.metadata.num_rows <= 0
        or parquet.metadata.num_row_groups <= 0
        or sum(row_group_rows) != parquet.metadata.num_rows
        or any(rows <= 0 for rows in row_group_rows)
    ):
        raise RuntimeError(f"Parquet row groups are incomplete: {path}")
    return {
        "path": str(path),
        "physical_bytes": path.stat().st_size,
        "physical_sha256": _sha256(path),
        "row_count": parquet.metadata.num_rows,
        "column_count": len(parquet.schema_arrow),
        "row_group_count": parquet.metadata.num_row_groups,
        "row_group_rows": row_group_rows,
        "schema_fingerprint": "sha256:"
        + hashlib.sha256(
            parquet.schema_arrow.serialize().to_pybytes()
        ).hexdigest(),
    }


def _checkpoint_fingerprint(
    checkpoint: Mapping[str, object],
) -> str:
    return canonical_semantic_fingerprint(
        {
            key: value
            for key, value in checkpoint.items()
            if key != "checkpoint_fingerprint"
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json_create_only(
    path: Path,
    value: Mapping[str, object],
) -> None:
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "BOUNDED_PIPELINE_VERSION",
    "build_bounded_final_from_spine",
]
