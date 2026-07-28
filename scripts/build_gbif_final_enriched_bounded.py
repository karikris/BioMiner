from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from biominer.gbif_final.bounded_pipeline import (
    build_bounded_final_from_spine,
)
from biominer.gbif_final.dimensions import (
    build_derived_assertion_dimension,
    build_species_enrichment_dimension,
)
from biominer.gbif_final.spine import (
    build_source_spine,
    validate_source_spine,
)
from biominer.gbif_final.telemetry import BoundedRunTelemetry


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    quality = args.quality_dir.resolve()
    state = args.state_dir.resolve()
    state.mkdir(parents=True, exist_ok=True)
    output = args.output_dir.resolve()
    telemetry = BoundedRunTelemetry(
        root_directory=(
            args.telemetry_dir.resolve()
            if args.telemetry_dir is not None
            else state / "telemetry"
        ),
        producer_git_sha=args.producer_git_sha,
        run_id=args.run_id,
        config={
            "temporal_parquet": args.temporal_parquet.resolve(),
            "pre_temporal_parquet": args.pre_temporal_parquet.resolve(),
            "temporal_audit": args.temporal_audit.resolve(),
            "registry_dir": args.registry_dir.resolve(),
            "source_assertions": (
                args.source_assertions.resolve()
                if args.source_assertions is not None
                else None
            ),
            "quality_dir": quality,
            "state_dir": state,
            "output_dir": output,
            "part_rows": args.part_rows,
            "batch_rows": args.batch_rows,
            "final_row_group_size": args.final_row_group_size,
            "threads": args.threads,
            "memory_limit": args.memory_limit,
            "peak_rss_target_bytes": args.peak_rss_target_bytes,
            "free_space_multiplier": args.free_space_multiplier,
            "minimum_headroom_bytes": args.minimum_headroom_bytes,
        },
        event_sink=lambda event: print(
            json.dumps(event, sort_keys=True),
            file=sys.stderr,
            flush=True,
        ),
    )
    output_was_present = output.is_dir()

    try:
        media_quality = (
            quality
            / "media_assertion_quality"
            / "media_assertion_quality.parquet"
        )
        occurrence_quality = (
            quality / "occurrence_quality" / "occurrence_quality.parquet"
        )
        rights_quality = (
            quality / "rights_and_attribution" / "media_rights.parquet"
        )
        duplicate_quality = (
            quality / "duplicates" / "duplicate_membership.parquet"
        )
        assertion_ledger = (
            quality
            / "quality_results"
            / "phase3"
            / "derived_assertions.parquet"
        )
        ai_readiness_parts = sorted(
            (quality / "ai_readiness" / "parts").glob("*.parquet")
        )
        if not ai_readiness_parts:
            raise FileNotFoundError(
                quality / "ai_readiness" / "parts" / "*.parquet"
            )

        source_spine = state / "source_spine"
        expected_spine_inputs = {
            "temporal": args.temporal_parquet,
            "pre_temporal": args.pre_temporal_parquet,
            "media_quality": media_quality,
            "temporal_audit": args.temporal_audit,
        }
        stage_started = time.monotonic()
        source_spine_was_present = source_spine.is_dir()
        _emit_stage(
            telemetry,
            event="stage_started",
            stage="source_spine",
            checkpoint_path=source_spine / "checkpoint.json",
        )
        if source_spine_was_present:
            spine_manifest = validate_source_spine(
                source_spine,
                expected_inputs=expected_spine_inputs,
            )
        else:
            spine_manifest = build_source_spine(
                temporal_parquet=args.temporal_parquet,
                pre_temporal_parquet=args.pre_temporal_parquet,
                media_quality_parquet=media_quality,
                temporal_audit_parquet=args.temporal_audit,
                output_directory=source_spine,
                producer_git_sha=args.producer_git_sha,
                part_rows=args.part_rows,
                batch_rows=args.batch_rows,
                verification_memory_limit=args.memory_limit,
                verification_threads=args.threads,
            )
        spine_rows = int(spine_manifest["counts"]["post_1960_rows"])
        _emit_stage(
            telemetry,
            event="stage_completed",
            stage="source_spine",
            checkpoint_path=source_spine / "manifest.json",
            rows_read=spine_rows,
            rows_written=0 if source_spine_was_present else spine_rows,
            rows_passed=spine_rows,
            rows_skipped_from_cache=(
                spine_rows if source_spine_was_present else 0
            ),
            elapsed_stage_time_seconds=time.monotonic() - stage_started,
        )

        dimensions = state / "dimensions"
        dimensions.mkdir(exist_ok=True)
        derived_dimension = dimensions / "derived_assertions.parquet"
        stage_started = time.monotonic()
        derived_was_present = derived_dimension.is_file()
        _emit_stage(
            telemetry,
            event="stage_started",
            stage="derived_assertion_dimension",
            checkpoint_path=derived_dimension.with_suffix(
                derived_dimension.suffix + ".receipt.json"
            ),
        )
        derived_receipt = build_derived_assertion_dimension(
            source_assertions=assertion_ledger,
            output_path=derived_dimension,
            producer_git_sha=args.producer_git_sha,
            batch_rows=args.batch_rows,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
        derived_rows = int(derived_receipt["artifact"]["row_count"])
        _emit_stage(
            telemetry,
            event="stage_completed",
            stage="derived_assertion_dimension",
            checkpoint_path=derived_dimension.with_suffix(
                derived_dimension.suffix + ".receipt.json"
            ),
            rows_read=derived_rows,
            rows_written=0 if derived_was_present else derived_rows,
            rows_passed=derived_rows,
            rows_skipped_from_cache=(
                derived_rows if derived_was_present else 0
            ),
            elapsed_stage_time_seconds=time.monotonic() - stage_started,
        )

        species_dimension = dimensions / "species_enrichments.parquet"
        stage_started = time.monotonic()
        species_was_present = species_dimension.is_file()
        _emit_stage(
            telemetry,
            event="stage_started",
            stage="species_enrichment_dimension",
            checkpoint_path=species_dimension.with_suffix(
                species_dimension.suffix + ".receipt.json"
            ),
        )
        species_receipt = build_species_enrichment_dimension(
            source_parquet=args.temporal_parquet,
            registry_dir=args.registry_dir,
            output_path=species_dimension,
            source_assertions_path=args.source_assertions,
            producer_git_sha=args.producer_git_sha,
            row_group_size=args.batch_rows,
        )
        species_rows = int(species_receipt["artifact"]["row_count"])
        _emit_stage(
            telemetry,
            event="stage_completed",
            stage="species_enrichment_dimension",
            checkpoint_path=species_dimension.with_suffix(
                species_dimension.suffix + ".receipt.json"
            ),
            rows_read=species_rows,
            rows_written=0 if species_was_present else species_rows,
            rows_passed=species_rows,
            rows_skipped_from_cache=(
                species_rows if species_was_present else 0
            ),
            elapsed_stage_time_seconds=time.monotonic() - stage_started,
        )

        manifest = build_bounded_final_from_spine(
            temporal_parquet=args.temporal_parquet,
            source_spine_directory=source_spine,
            media_quality_parquet=media_quality,
            occurrence_quality_parquet=occurrence_quality,
            rights_quality_parquet=rights_quality,
            duplicate_quality_parquet=duplicate_quality,
            ai_readiness_parts=ai_readiness_parts,
            derived_assertion_dimension=derived_dimension,
            species_enrichment_dimension=species_dimension,
            work_directory=state / "pipeline",
            output_directory=output,
            producer_git_sha=args.producer_git_sha,
            threads=args.threads,
            memory_limit=args.memory_limit,
            batch_rows=args.batch_rows,
            final_row_group_size=args.final_row_group_size,
            free_space_multiplier=args.free_space_multiplier,
            minimum_headroom_bytes=args.minimum_headroom_bytes,
            progress=telemetry.emit_payload,
        )
        run_receipt = telemetry.finish(
            output_manifest=output / "manifest.json",
            rows=int(manifest["counts"]["rows"]),
            resumed_output=output_was_present,
        )
    except BaseException as error:
        try:
            telemetry.fail(error, stage="orchestration")
        except Exception:
            pass
        raise

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "telemetry_receipt": str(
                    telemetry.invocation_directory / "run_receipt.json"
                ),
                "receipt_fingerprint": run_receipt[
                    "receipt_fingerprint"
                ],
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 0


def _emit_stage(
    telemetry: BoundedRunTelemetry,
    *,
    event: str,
    stage: str,
    checkpoint_path: Path,
    rows_read: int = 0,
    rows_written: int = 0,
    rows_passed: int = 0,
    rows_skipped_from_cache: int = 0,
    elapsed_stage_time_seconds: float | None = None,
) -> None:
    telemetry.emit(
        event,
        stage=stage,
        partition=None,
        rows_read=rows_read,
        rows_written=rows_written,
        rows_passed=rows_passed,
        rows_failed=0,
        rows_unresolved=0,
        rows_skipped_from_cache=rows_skipped_from_cache,
        current_provider=None,
        current_host=None,
        requests_completed=0,
        retries=0,
        rate_limit_events=0,
        bytes_downloaded=0,
        network_scope="NOT_APPLICABLE",
        elapsed_stage_time_seconds=elapsed_stage_time_seconds,
        rows_per_second=(
            rows_passed / max(elapsed_stage_time_seconds, 1e-9)
            if elapsed_stage_time_seconds is not None
            else None
        ),
        estimated_work_remaining=0 if event.endswith("completed") else None,
        checkpoint_path=str(checkpoint_path),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the post-1960 legacy GBIF final dataset through "
            "checksum-bound, restartable source-ordinal windows."
        )
    )
    parser.add_argument("--temporal-parquet", type=Path, required=True)
    parser.add_argument("--pre-temporal-parquet", type=Path, required=True)
    parser.add_argument("--temporal-audit", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--source-assertions", type=Path)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--telemetry-dir", type=Path)
    parser.add_argument("--part-rows", type=int, default=250_000)
    parser.add_argument("--batch-rows", type=int, default=65_536)
    parser.add_argument("--final-row-group-size", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument(
        "--peak-rss-target-bytes",
        type=int,
        default=16 * 1024**3,
    )
    parser.add_argument("--free-space-multiplier", type=float, default=1.25)
    parser.add_argument(
        "--minimum-headroom-bytes",
        type=int,
        default=2 * 1024**3,
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
