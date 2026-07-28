from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from biominer.config import create_workstore, load_biominer_config
from biominer.gbif_media_resolution.pipeline import (
    finalize_resolution,
    prepare_resolution,
    publish_v4,
    run_worker,
)
from biominer.gbif_media_resolution.pilot_audit import (
    publish_pilot_execution_audit,
    write_pilot_execution_review,
)
from biominer.gbif_media_resolution.resolver import MediaURLResolver, ResolverConfig
from biominer.workstore.base import WorkStore
from biominer.workstore.sqlite import SQLiteWorkStore


COMMAND = "gbif-media-url-resolve"


def add_gbif_media_resolution_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    command = subparsers.add_parser(
        COMMAND,
        help="recover and publish missing direct GBIF multimedia URLs",
    )
    stages = command.add_subparsers(dest="gbif_media_url_command")

    prepare = stages.add_parser("prepare", help="validate input and enqueue source rows")
    _add_workstore_arguments(prepare)
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--source-manifest", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--expected-missing-rows", type=int, default=130_689)
    prepare.add_argument("--expected-rights-blocked-rows", type=int, default=4_055)
    prepare.add_argument("--enqueue-batch-rows", type=int, default=1_000)
    prepare.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    prepare.add_argument(
        "--allow-full-queue",
        action="store_true",
        help="explicitly allow enqueueing the full resolver workload",
    )
    prepare.add_argument(
        "--pilot-acceptance-manifest",
        help="required PASS execution-audit manifest before full-queue preparation",
    )
    _add_resolver_arguments(prepare)

    worker = stages.add_parser("work", help="resolve one or more bounded batches")
    _add_workstore_arguments(worker)
    worker.add_argument("--output-root", required=True)
    worker.add_argument("--run-id", required=True)
    worker.add_argument("--worker-id", required=True)
    worker.add_argument("--batch-rows", type=int, default=25)
    worker.add_argument("--max-batches", type=int, default=1)
    worker.add_argument("--stale-after-seconds", type=int, default=900)
    worker.add_argument(
        "--execute-network",
        action="store_true",
        help="explicit opt-in required before any resolver network requests",
    )
    _add_resolver_arguments(worker)

    finalize = stages.add_parser("finalize", help="reduce shards and publish v1 sidecars")
    _add_workstore_arguments(finalize)
    finalize.add_argument("--output-root", required=True)
    finalize.add_argument("--output-directory", required=True)
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--expected-rows", type=int)

    review = stages.add_parser(
        "prepare-review",
        help="write a create-only manual review table from finalized pilot results",
    )
    review.add_argument("--pilot-selection", required=True)
    review.add_argument("--resolution-directory", required=True)
    review.add_argument("--output", required=True)

    audit_pilot = stages.add_parser(
        "audit-pilot",
        help="publish the checksum-bound post-execution pilot audit",
    )
    audit_pilot.add_argument("--prepare-receipt", required=True)
    audit_pilot.add_argument("--pilot-selection", required=True)
    audit_pilot.add_argument("--resolution-directory", required=True)
    audit_pilot.add_argument("--reviewed-pilot", required=True)
    audit_pilot.add_argument("--output-directory", required=True)
    audit_pilot.add_argument("--expected-rows", type=int, default=823)
    audit_pilot.add_argument("--code-commit", required=True)
    audit_pilot.add_argument("--adapter-test-receipt", required=True)

    publish = stages.add_parser("publish-v4", help="publish v4 Parquet and indexed DuckDB")
    publish.add_argument("--source", required=True)
    publish.add_argument("--source-manifest", required=True)
    publish.add_argument("--resolution-directory", required=True)
    publish.add_argument("--output-directory", required=True)
    publish.add_argument("--batch-rows", type=int, default=50_000)
    publish.add_argument("--duckdb-memory-limit", default="24GB")
    publish.add_argument("--duckdb-threads", type=int, default=8)


def run_gbif_media_resolution_command(args: argparse.Namespace) -> int:
    stage = args.gbif_media_url_command
    if stage is None:
        return 2
    if stage == "prepare":
        if args.mode == "full" and not args.allow_full_queue:
            raise ValueError("full queue preparation requires --allow-full-queue")
        if args.mode == "full" and not args.pilot_acceptance_manifest:
            raise ValueError(
                "full queue preparation requires a PASS pilot acceptance manifest"
            )
        resolver_config = _resolver_config(args)
        result = prepare_resolution(
            source=args.source,
            source_manifest=args.source_manifest,
            output_root=args.output_root,
            workstore=_workstore(args),
            run_id=args.run_id,
            expected_missing_rows=args.expected_missing_rows,
            enqueue_batch_rows=args.enqueue_batch_rows,
            mode=args.mode,
            expected_rights_blocked_rows=args.expected_rights_blocked_rows,
            resolver_config=resolver_config,
            pilot_acceptance_manifest=args.pilot_acceptance_manifest,
        )
    elif stage == "work":
        if not args.execute_network:
            raise ValueError("resolver network execution requires --execute-network")
        if args.max_batches <= 0:
            raise ValueError("max_batches must be positive")
        store = _workstore(args)
        config = _resolver_config(args)
        batches: list[dict[str, Any]] = []
        with MediaURLResolver(
            config=config,
            request_guard=lambda host: store.publication_lock(
                f"gbif_media_url_resolution:origin:{host}"
            ),
        ) as resolver:
            for _ in range(args.max_batches):
                receipt = run_worker(
                    workstore=store,
                    output_root=args.output_root,
                    run_id=args.run_id,
                    worker_id=args.worker_id,
                    batch_rows=args.batch_rows,
                    stale_after_seconds=args.stale_after_seconds,
                    resolver=resolver,
                )
                batches.append(receipt)
                _event("gbif_media_url_worker_batch", **receipt)
                if receipt["claimed_rows"] == 0:
                    break
        result = {
            "run_id": args.run_id,
            "worker_id": args.worker_id,
            "batches": len(batches),
            "claimed_rows": sum(int(item["claimed_rows"]) for item in batches),
            "completed_rows": sum(int(item["completed_rows"]) for item in batches),
            "attempt_rows": sum(int(item.get("attempt_rows", 0)) for item in batches),
        }
    elif stage == "finalize":
        result = finalize_resolution(
            workstore=_workstore(args),
            run_id=args.run_id,
            output_root=args.output_root,
            output_directory=args.output_directory,
            expected_rows=args.expected_rows,
        )
    elif stage == "prepare-review":
        result = write_pilot_execution_review(
            pilot_selection=args.pilot_selection,
            resolution_results=(
                Path(args.resolution_directory) / "resolution_results.parquet"
            ),
            output_path=args.output,
        )
    elif stage == "audit-pilot":
        result = publish_pilot_execution_audit(
            prepare_receipt=args.prepare_receipt,
            pilot_selection=args.pilot_selection,
            resolution_directory=args.resolution_directory,
            reviewed_pilot=args.reviewed_pilot,
            output_directory=args.output_directory,
            expected_rows=args.expected_rows,
            code_commit=args.code_commit,
            adapter_test_receipt=args.adapter_test_receipt,
        )
    elif stage == "publish-v4":
        result = publish_v4(
            source=args.source,
            source_manifest=args.source_manifest,
            resolution_directory=args.resolution_directory,
            output_directory=args.output_directory,
            batch_rows=args.batch_rows,
            duckdb_memory_limit=args.duckdb_memory_limit,
            duckdb_threads=args.duckdb_threads,
        )
    else:  # pragma: no cover - argparse contract guard.
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _add_workstore_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sqlite-workstore",
        help="explicit local/test SQLite state path; production uses configured PostgreSQL",
    )


def _add_resolver_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--max-redirects", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--minimum-origin-interval-seconds", type=float, default=0.0)


def _resolver_config(args: argparse.Namespace) -> ResolverConfig:
    return ResolverConfig(
        max_attempts=args.max_attempts,
        max_redirects=args.max_redirects,
        timeout_seconds=args.timeout_seconds,
        minimum_origin_interval_seconds=args.minimum_origin_interval_seconds,
    )


def _workstore(args: argparse.Namespace) -> WorkStore:
    sqlite_path = getattr(args, "sqlite_workstore", None)
    if sqlite_path:
        return SQLiteWorkStore(Path(sqlite_path))
    config = load_biominer_config(getattr(args, "config", None))
    return create_workstore(config.workstore)


def _event(name: str, **values: Any) -> None:
    print(json.dumps({"event": name, **values}, sort_keys=True), flush=True)
