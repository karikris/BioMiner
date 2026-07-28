from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    quality = args.quality_dir.resolve()
    state = args.state_dir.resolve()
    state.mkdir(parents=True, exist_ok=True)

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
    if source_spine.exists():
        validate_source_spine(
            source_spine,
            expected_inputs=expected_spine_inputs,
        )
    else:
        build_source_spine(
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

    dimensions = state / "dimensions"
    dimensions.mkdir(exist_ok=True)
    derived_dimension = dimensions / "derived_assertions.parquet"
    build_derived_assertion_dimension(
        source_assertions=assertion_ledger,
        output_path=derived_dimension,
        producer_git_sha=args.producer_git_sha,
        batch_rows=args.batch_rows,
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    species_dimension = dimensions / "species_enrichments.parquet"
    build_species_enrichment_dimension(
        source_parquet=args.temporal_parquet,
        registry_dir=args.registry_dir,
        output_path=species_dimension,
        source_assertions_path=args.source_assertions,
        producer_git_sha=args.producer_git_sha,
        row_group_size=args.batch_rows,
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
        output_directory=args.output_dir,
        producer_git_sha=args.producer_git_sha,
        threads=args.threads,
        memory_limit=args.memory_limit,
        batch_rows=args.batch_rows,
        final_row_group_size=args.final_row_group_size,
        free_space_multiplier=args.free_space_multiplier,
        minimum_headroom_bytes=args.minimum_headroom_bytes,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


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
    parser.add_argument("--part-rows", type=int, default=250_000)
    parser.add_argument("--batch-rows", type=int, default=65_536)
    parser.add_argument("--final-row-group-size", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--free-space-multiplier", type=float, default=1.25)
    parser.add_argument(
        "--minimum-headroom-bytes",
        type=int,
        default=2 * 1024**3,
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
