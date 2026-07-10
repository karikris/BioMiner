#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.config import ConfigError, create_storage_backend, load_biominer_config, redact_config
from biominer.run.paths import RunArtifactUris
from biominer.secrets_loader import load_runtime_secrets_env
from biominer.storage.uri import join_uri


DEFAULT_RUN_PREFIX = "s3://biominer/biominer/runs/papilio_demoleus"
DEFAULT_RUN_ID = "papilio_demoleus_ranked_slices_20260708t110046z"
DEFAULT_ARTIFACT = "source_records"


@dataclass(frozen=True)
class PulledShard:
    source_uri: str
    local_path: str | None
    row_count: int | None
    status: str
    error: str | None = None


def main() -> None:
    raise SystemExit(run(parse_args()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull Flickr metadata/evidence Parquet from BioMiner S3 storage into a local analysis directory. "
            "This does not download Flickr images or raw API JSON."
        )
    )
    parser.add_argument("--config", default="config/biominer.cloud.example.toml")
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--remote-prefix", help="Explicit S3 prefix to list. Overrides --run-prefix/--run-id/--artifact/--stage.")
    parser.add_argument("--artifact", choices=("source_records", "object_detections", "object_scores", "object_evidence", "photo_summary"))
    parser.add_argument("--stage", help="Staging evidence shard stage to list when --artifact is not set.")
    parser.add_argument("--output-dir", default="runs/analysis/flickr_s3_pull")
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--compact-output", help="Optional local Parquet path to concatenate pulled shards.")
    parser.add_argument("--dry-run", action="store_true", help="List matching S3 Parquet objects without downloading them.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    load_runtime_secrets_env()
    try:
        config = load_biominer_config(args.config)
        storage = create_storage_backend(config.storage)
    except (ConfigError, FileNotFoundError) as exc:
        print(json.dumps({"error": str(exc), "config": str(args.config)}, indent=2, sort_keys=True))
        return 2
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    uris = _target_uris(args, storage)
    if args.limit_shards and args.limit_shards > 0:
        uris = uris[: args.limit_shards]

    pulled: list[PulledShard] = []
    frames: list[pl.DataFrame] = []
    for index, uri in enumerate(uris):
        if args.dry_run:
            pulled.append(PulledShard(source_uri=uri, local_path=None, row_count=None, status="listed"))
            continue
        try:
            frame = storage.read_parquet(uri)
            local_path = output_dir / "shards" / f"part-{index:06d}.parquet"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(local_path, compression="zstd")
            frames.append(frame)
            pulled.append(PulledShard(source_uri=uri, local_path=str(local_path), row_count=frame.height, status="pulled"))
        except Exception as exc:  # noqa: BLE001 - report per-shard failures without leaking config.
            pulled.append(PulledShard(source_uri=uri, local_path=None, row_count=None, status="failed", error=str(exc)))

    compact_path = None
    if args.compact_output and not args.dry_run and frames:
        compact = pl.concat(frames, how="diagonal_relaxed")
        compact_path = Path(args.compact_output)
        compact_path.parent.mkdir(parents=True, exist_ok=True)
        compact.write_parquet(compact_path, compression="zstd")

    manifest = {
        "config": redact_config(config),
        "target": _target_description(args),
        "dry_run": bool(args.dry_run),
        "matched_uris": len(uris),
        "pulled_shards": [asdict(item) for item in pulled],
        "rows_pulled": sum(item.row_count or 0 for item in pulled),
        "compact_output": str(compact_path) if compact_path else None,
        "notes": [
            "Flickr image bytes are not downloaded by this script.",
            "Raw Flickr API JSON is not downloaded by this script.",
            "Local Parquet outputs are zstd-compressed.",
        ],
    }
    manifest_path = output_dir / "pull_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **{k: manifest[k] for k in ("dry_run", "matched_uris", "rows_pulled", "compact_output")}}, indent=2, sort_keys=True))
    return 0 if all(item.status != "failed" for item in pulled) else 2


def _target_uris(args: argparse.Namespace, storage: Any) -> list[str]:
    if args.remote_prefix:
        return storage.list_shards(args.remote_prefix)
    if args.artifact:
        uri = _artifact_uri(args, args.artifact)
        return [uri] if storage.exists(uri) else []
    if not args.stage:
        uri = _artifact_uri(args, DEFAULT_ARTIFACT)
        return [uri] if storage.exists(uri) else []
    prefix = join_uri(_run_artifacts(args).staging_uri, "evidence", f"stage={args.stage}")
    return storage.list_shards(prefix)


def _artifact_uri(args: argparse.Namespace, artifact: str) -> str:
    artifacts = _run_artifacts(args)
    return {
        "source_records": artifacts.source_records_uri,
        "object_detections": artifacts.object_detections_uri,
        "object_scores": artifacts.object_scores_uri,
        "object_evidence": artifacts.object_evidence_uri,
        "photo_summary": artifacts.photo_summary_uri,
    }[artifact]


def _run_artifacts(args: argparse.Namespace) -> RunArtifactUris:
    return RunArtifactUris.from_prefix(args.run_prefix, run_id=args.run_id)


def _target_description(args: argparse.Namespace) -> dict[str, str | None]:
    target = {
        "run_prefix": args.run_prefix,
        "run_id": args.run_id,
        "remote_prefix": args.remote_prefix,
        "artifact": args.artifact,
        "stage": None if args.artifact else args.stage,
    }
    if args.remote_prefix:
        target["resolved_target"] = "remote_prefix"
    elif args.artifact:
        target["resolved_target"] = "artifact"
        target["resolved_artifact"] = args.artifact
    elif args.stage:
        target["resolved_target"] = "stage"
    else:
        target["resolved_target"] = "artifact"
        target["resolved_artifact"] = DEFAULT_ARTIFACT
    return target


if __name__ == "__main__":
    main()
