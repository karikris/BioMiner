from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

from biominer.gbif_quality.acceptance import publish_acceptance_audit
from biominer.gbif_quality.ai_readiness import publish_ai_readiness
from biominer.gbif_quality.concentration import publish_concentration_metrics
from biominer.gbif_quality.duplicates import publish_duplicate_groups
from biominer.gbif_quality.final_reports import publish_final_reports
from biominer.gbif_quality.freshness import publish_freshness_audit
from biominer.gbif_quality.gates import publish_gate_breakdowns
from biominer.gbif_quality.incremental import publish_incremental_state
from biominer.gbif_quality.media_resources import publish_media_resources
from biominer.gbif_quality.provider_enrichment import (
    publish_provider_enrichment_registry,
)
from biominer.gbif_quality.representativeness import publish_representativeness
from biominer.gbif_quality.review_capsules import publish_review_capsules
from biominer.gbif_quality.rights import publish_media_rights
from biominer.gbif_quality.source_lineage import publish_source_assertion_lineage

from biominer.gbif_quality.pipeline import (
    Phase1Config,
    Phase2Config,
    Phase3Config,
    run_phase1_baseline,
    run_phase2_local_checks,
    run_phase3_enrichment,
)


COMMAND = "gbif-media-quality"
DEFAULT_DATA_ROOT = "data/derived/gbif_media_database/v4"
DEFAULT_REPORT_ROOT = "reports/gbif_media_database/v4"
DEFAULT_EXPECTED_ROWS = 16_612_063
DEFAULT_V3_SHA256 = (
    "c96505f410723da57db4bd11bcffdc4e72be59ee59ecbaad8f4af8677229e57f"
)


def add_gbif_quality_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    command = subparsers.add_parser(
        COMMAND,
        help="audit and enrich the pinned GBIF media database",
    )
    stages = command.add_subparsers(dest="gbif_quality_command")
    baseline = stages.add_parser(
        "baseline", help="publish the reconciled Phase 1 quality baseline"
    )
    baseline.add_argument("--repository-root", default=".")
    baseline.add_argument("--data-output", default=DEFAULT_DATA_ROOT)
    baseline.add_argument("--report-output", default=DEFAULT_REPORT_ROOT)
    baseline.add_argument("--temp-directory")
    baseline.add_argument("--memory-limit", default="4GB")
    baseline.add_argument("--occurrence-batch-size", type=int, default=8)
    local = stages.add_parser(
        "local-checks", help="run or resume the request-free Phase 2 checks"
    )
    local.add_argument("--repository-root", default=".")
    local.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    local.add_argument("--temp-directory")
    local.add_argument("--memory-limit", default="4GB")
    local.add_argument("--threads", type=int, default=4)
    local.add_argument("--batch-rows", type=int, default=100_000)
    enrichment = stages.add_parser(
        "enrich", help="run or resume deterministic Phase 3 enrichment"
    )
    enrichment.add_argument("--repository-root", default=".")
    enrichment.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    enrichment.add_argument("--memory-limit", default="4GB")
    enrichment.add_argument("--threads", type=int, default=4)
    enrichment.add_argument("--batch-rows", type=int, default=50_000)

    for name, help_text in (
        ("rights", "publish normalized media rights and attribution evidence"),
        ("duplicates", "publish URL and metadata duplicate groups"),
        ("ai-readiness", "publish fail-closed AI-readiness decisions"),
        ("representativeness", "publish coverage and provider bias scorecards"),
        ("concentration", "publish provider, creator, regional, and temporal concentration"),
        ("media-resources", "publish canonical media-resource observations"),
        ("gates", "publish the seven completeness-gate breakdowns"),
        ("review-capsules", "publish deterministic manual-review capsules"),
        ("incremental", "publish source-domain hashes and a changed-row queue"),
        ("freshness", "classify provider datasets and derived snapshot freshness"),
    ):
        stage = stages.add_parser(name, help=help_text)
        _add_publication_arguments(stage)
        if name == "rights":
            stage.add_argument("--batch-rows", type=int, default=50_000)
        elif name in {"media-resources", "incremental"}:
            stage.add_argument("--partitions", type=int, default=16)
        elif name == "review-capsules":
            stage.add_argument("--sample-seed", default="gbif-media-v4-review-v1")
            stage.add_argument("--max-per-stratum", type=int, default=10)
        if name == "incremental":
            stage.add_argument("--previous-state-glob")
        if name == "freshness":
            stage.add_argument("--provider-stale-days", type=int, default=365)
            stage.add_argument("--derived-stale-days", type=int, default=30)

    lineage = stages.add_parser(
        "source-lineage", help="publish source-row locations and cryptographic value hashes"
    )
    lineage.add_argument("--repository-root", default=".")
    lineage.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    lineage.add_argument("--multimedia-parquet")
    lineage.add_argument(
        "--output-directory",
        default=f"{DEFAULT_DATA_ROOT}/source_lineage/identity_v2",
    )
    lineage.add_argument("--expected-rows", type=int, default=18_680_565)
    lineage.add_argument("--partition-rows", type=int, default=1_000_000)
    lineage.add_argument("--code-commit")
    lineage.add_argument("--temp-directory")
    lineage.add_argument("--memory-limit", default="6GB")
    lineage.add_argument("--threads", type=int, default=4)

    provider_registry = stages.add_parser(
        "provider-registry", help="publish the prioritized provider enrichment interfaces"
    )
    provider_registry.add_argument("--repository-root", default=".")
    provider_registry.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    provider_registry.add_argument(
        "--output-directory", default=f"{DEFAULT_DATA_ROOT}/provider_enrichment"
    )
    provider_registry.add_argument("--source-snapshot-id")
    provider_registry.add_argument("--code-commit")

    reports = stages.add_parser("reports", help="render the final evidence report suite")
    reports.add_argument("--repository-root", default=".")
    reports.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    reports.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    reports.add_argument("--code-commit")

    acceptance = stages.add_parser(
        "acceptance", help="publish the executable 42-criterion acceptance audit"
    )
    acceptance.add_argument("--repository-root", default=".")
    acceptance.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    acceptance.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    acceptance.add_argument(
        "--output-directory",
        default=f"{DEFAULT_DATA_ROOT}/quality_results/global_acceptance",
    )
    acceptance.add_argument(
        "--test-receipt",
        default=f"{DEFAULT_DATA_ROOT}/quality_results/test_receipt.json",
    )
    acceptance.add_argument("--expected-v3-sha256", default=DEFAULT_V3_SHA256)
    acceptance.add_argument("--code-commit")


def run_gbif_quality_command(args: argparse.Namespace) -> int:
    if args.gbif_quality_command in {
        "rights",
        "duplicates",
        "ai-readiness",
        "representativeness",
        "concentration",
        "media-resources",
        "gates",
        "review-capsules",
        "incremental",
        "freshness",
        "source-lineage",
        "provider-registry",
        "reports",
        "acceptance",
    }:
        result = _run_publication(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.gbif_quality_command == "enrich":
        result = run_phase3_enrichment(
            Phase3Config(
                repository_root=args.repository_root,
                data_root=args.data_root,
                memory_limit=args.memory_limit,
                threads=args.threads,
                batch_rows=args.batch_rows,
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.gbif_quality_command == "local-checks":
        result = run_phase2_local_checks(
            Phase2Config(
                repository_root=args.repository_root,
                data_root=args.data_root,
                temp_directory=args.temp_directory,
                memory_limit=args.memory_limit,
                threads=args.threads,
                batch_rows=args.batch_rows,
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.gbif_quality_command != "baseline":
        return 2
    result = run_phase1_baseline(
        Phase1Config(
            repository_root=args.repository_root,
            data_output=args.data_output,
            report_output=args.report_output,
            temp_directory=args.temp_directory,
            memory_limit=args.memory_limit,
            occurrence_batch_size=args.occurrence_batch_size,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _add_publication_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--v3")
    parser.add_argument("--output-directory")
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS)
    parser.add_argument("--source-snapshot-id")
    parser.add_argument("--code-commit")
    parser.add_argument("--temp-directory")
    parser.add_argument("--memory-limit", default="6GB")
    parser.add_argument("--threads", type=int, default=4)


def _run_publication(args: argparse.Namespace) -> dict[str, object]:
    repository = Path(args.repository_root).resolve()
    data = _resolve_from_repository(repository, args.data_root)
    commit = args.code_commit or _git_commit(repository)
    stage = args.gbif_quality_command
    if stage == "provider-registry":
        return publish_provider_enrichment_registry(
            output_directory=_resolve_from_repository(repository, args.output_directory),
            source_snapshot_id=args.source_snapshot_id or _source_snapshot_id(data),
            code_commit=commit,
        )
    if stage == "source-lineage":
        multimedia = (
            _resolve_from_repository(repository, args.multimedia_parquet)
            if args.multimedia_parquet
            else _artifact_from_inventory(repository, data, "multimedia_extension")
        )
        return publish_source_assertion_lineage(
            multimedia_parquet=multimedia,
            source_status_parquet=data / "source_lineage/source_media_status.parquet",
            source_inventory_json=data / "source_inventory.json",
            output_directory=_resolve_from_repository(repository, args.output_directory),
            expected_rows=args.expected_rows,
            code_commit=commit,
            partition_rows=args.partition_rows,
            memory_limit=args.memory_limit,
            threads=args.threads,
            temp_directory=(
                _resolve_from_repository(repository, args.temp_directory)
                if args.temp_directory
                else None
            ),
        )
    if stage == "reports":
        return publish_final_reports(
            data_root=data,
            report_root=_resolve_from_repository(repository, args.report_root),
            code_commit=commit,
        )
    if stage == "acceptance":
        return publish_acceptance_audit(
            repository_root=repository,
            data_root=data,
            report_root=_resolve_from_repository(repository, args.report_root),
            output_directory=_resolve_from_repository(repository, args.output_directory),
            test_receipt=_resolve_from_repository(repository, args.test_receipt),
            expected_v3_sha256=args.expected_v3_sha256,
            code_commit=commit,
        )

    v3 = (
        _resolve_from_repository(repository, args.v3)
        if args.v3
        else _v3_from_inventory(repository, data)
    )
    snapshot = args.source_snapshot_id or _source_snapshot_id(data)
    output = _resolve_from_repository(
        repository, args.output_directory or str(data / _default_output_name(stage))
    )
    common = {
        "v3_parquet": v3,
        "output_directory": output,
        "source_snapshot_id": snapshot,
        "expected_rows": args.expected_rows,
        "code_commit": commit,
    }
    media_quality = data / "media_assertion_quality/media_assertion_quality.parquet"
    duplicates = data / "duplicates/duplicate_membership.parquet"
    ai_glob = data / "ai_readiness/parts/*.parquet"
    bounded = {
        "memory_limit": args.memory_limit,
        "threads": args.threads,
        "temp_directory": (
            _resolve_from_repository(repository, args.temp_directory)
            if args.temp_directory
            else None
        ),
    }
    if stage == "freshness":
        return publish_freshness_audit(
            v3_parquet=v3,
            source_inventory_json=data / "source_inventory.json",
            data_root=data,
            output_directory=output,
            expected_rows=args.expected_rows,
            code_commit=commit,
            provider_stale_days=args.provider_stale_days,
            derived_stale_days=args.derived_stale_days,
            **bounded,
        )
    if stage == "rights":
        return publish_media_rights(
            **common,
            media_quality_parquet=media_quality,
            batch_rows=args.batch_rows,
        )
    if stage == "duplicates":
        return publish_duplicate_groups(
            **common, media_quality_parquet=media_quality, **bounded
        )
    if stage == "ai-readiness":
        return publish_ai_readiness(
            **common,
            media_quality_parquet=media_quality,
            occurrence_quality_parquet=data / "occurrence_quality/occurrence_quality.parquet",
            rights_parquet=data / "rights_and_attribution/media_rights.parquet",
            duplicates_parquet=duplicates,
            taxonomy_repairs_parquet=data / "derived_assertions/taxonomy/species_rank_repairs.parquet",
            **bounded,
        )
    if stage == "representativeness":
        return publish_representativeness(
            **common,
            media_quality_parquet=media_quality,
            ai_readiness_glob=ai_glob,
            **bounded,
        )
    if stage == "concentration":
        return publish_concentration_metrics(
            **common,
            media_quality_parquet=media_quality,
            ai_readiness_glob=ai_glob,
            **bounded,
        )
    if stage == "media-resources":
        resource_common = dict(common)
        resource_common.pop("v3_parquet")
        resource_common["expected_assertion_rows"] = resource_common.pop("expected_rows")
        return publish_media_resources(
            **resource_common,
            duplicates_parquet=duplicates,
            partitions=args.partitions,
            **bounded,
        )
    if stage == "gates":
        return publish_gate_breakdowns(
            **common,
            media_quality_parquet=media_quality,
            ai_readiness_glob=ai_glob,
            **bounded,
        )
    if stage == "review-capsules":
        return publish_review_capsules(
            **common,
            media_quality_parquet=media_quality,
            rights_parquet=data / "rights_and_attribution/media_rights.parquet",
            duplicates_parquet=duplicates,
            ai_readiness_glob=ai_glob,
            sample_seed=args.sample_seed,
            max_per_stratum=args.max_per_stratum,
            **bounded,
        )
    if stage == "incremental":
        return publish_incremental_state(
            **common,
            media_quality_parquet=media_quality,
            duplicates_parquet=duplicates,
            previous_state_glob=(
                str(_resolve_from_repository(repository, args.previous_state_glob))
                if args.previous_state_glob
                else None
            ),
            partitions=args.partitions,
            **bounded,
        )
    raise ValueError(f"unsupported GBIF quality publication stage: {stage}")


def _resolve_from_repository(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def _source_snapshot_id(data: Path) -> str:
    value = json.loads((data / "manifest.json").read_text()).get("source_snapshot_id")
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError("v4 manifest has no valid source_snapshot_id")
    return value


def _v3_from_inventory(repository: Path, data: Path) -> Path:
    return _artifact_from_inventory(repository, data, "rights_filtered_v3")


def _artifact_from_inventory(repository: Path, data: Path, role: str) -> Path:
    rows = pq.read_table(
        data / "source_inventory.parquet",
        filters=[("artifact_role", "=", role)],
    ).to_pylist()
    if len(rows) != 1:
        raise ValueError(f"source inventory must contain exactly one {role} row")
    return _resolve_from_repository(repository, rows[0]["path"])


def _git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode or not result.stdout.strip():
        raise ValueError(f"cannot resolve Git commit: {result.stderr.strip()}")
    return result.stdout.strip()


def _default_output_name(stage: str) -> str:
    return {
        "rights": "rights_and_attribution",
        "duplicates": "duplicates",
        "ai-readiness": "ai_readiness",
        "representativeness": "representativeness",
        "concentration": "representativeness_concentration",
        "media-resources": "media_resources",
        "gates": "completeness_gates",
        "review-capsules": "quality_results/review_capsules",
        "incremental": "incremental_state",
        "freshness": "freshness",
    }[stage]


__all__ = ["COMMAND", "add_gbif_quality_parser", "run_gbif_quality_command"]
