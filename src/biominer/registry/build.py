from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import subprocess
from typing import Any

from biominer.registry.compiler import compile_registry_fixture
from biominer.registry.gbif_production import ProductionGBIFClient
from biominer.registry.gbif_source import build_gbif_source_snapshot
from biominer.registry.scope import load_scope


logger = logging.getLogger(__name__)


def build_registry(
    *,
    output_dir: str | Path,
    registry_version: str,
    scope_path: str | Path = "config/butterfly_scope.json",
    source_json: str | Path | None = None,
    reuse_source_json: bool = False,
    report_dir: str | Path = "reports",
    retrieved_at: str | None = None,
    workers: int = 8,
    progress_every: int = 100,
    checkpoint_every: int = 500,
    max_retries: int = 5,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_json) if source_json else output / "gbif_source_snapshot.json"
    retrieved = retrieved_at or datetime.now(UTC).isoformat()
    logger.info(
        "registry.build.start version=%s output=%s workers=%d progress_every=%d checkpoint_every=%d max_retries=%d",
        registry_version,
        output,
        workers,
        progress_every,
        checkpoint_every,
        max_retries,
    )

    if reuse_source_json:
        if not source_path.exists():
            raise FileNotFoundError(f"--reuse-source-json requires an existing source JSON: {source_path}")
    else:
        with ProductionGBIFClient(max_retries=max_retries, max_connections=workers) as client:
            snapshot = build_gbif_source_snapshot(
                client,
                load_scope(scope_path),
                retrieved_at=retrieved,
                checkpoint_dir=output / "checkpoints",
                workers=workers,
                progress_every=progress_every,
                checkpoint_every=checkpoint_every,
                max_retries=max_retries,
                client_factory=lambda: ProductionGBIFClient(max_retries=max_retries, max_connections=workers),
            )
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("registry.build.compile.start source=%s", source_path)
    manifest = compile_registry_fixture(
        source_path,
        output,
        registry_version=registry_version,
        scope_path=scope_path,
    )
    logger.info(
        "registry.build.compile.complete status=%s taxa=%s names=%s queries=%s",
        manifest.get("qa_status"),
        manifest.get("taxa_rows"),
        manifest.get("name_rows"),
        manifest.get("query_definition_rows"),
    )
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    report = _build_report(
        manifest=manifest,
        source_payload=source_payload,
        source_path=source_path,
        output_dir=output,
        registry_version=registry_version,
        retrieved_at=retrieved,
    )
    report_paths = _write_reports(report, report_dir=Path(report_dir), registry_version=registry_version)
    logger.info("registry.build.complete version=%s status=%s output=%s", registry_version, manifest.get("qa_status"), output)
    return {
        "registry_version": registry_version,
        "source_json": str(source_path),
        "output_dir": str(output),
        "manifest": manifest,
        **report_paths,
    }


def _build_report(
    *,
    manifest: dict[str, Any],
    source_payload: dict[str, Any],
    source_path: Path,
    output_dir: Path,
    registry_version: str,
    retrieved_at: str,
) -> dict[str, Any]:
    return {
        "command": "biominer registry build",
        "git_sha": _git_sha(),
        "registry_version": registry_version,
        "status": manifest.get("qa_status"),
        "start": retrieved_at,
        "end": datetime.now(UTC).isoformat(),
        "source_json": str(source_path),
        "output_dir": str(output_dir),
        "source": source_payload.get("source"),
        "source_version": source_payload.get("source_version"),
        "api_calls_used": source_payload.get("metrics", {}).get("gbif_calls"),
        "api_request_attempts": source_payload.get("metrics", {}).get("gbif_request_attempts"),
        "api_retries": source_payload.get("metrics", {}).get("gbif_retries"),
        "workers": source_payload.get("metrics", {}).get("workers"),
        "checkpoint_every": source_payload.get("metrics", {}).get("checkpoint_every"),
        "progress_every": source_payload.get("metrics", {}).get("progress_every"),
        "resumed_species": source_payload.get("metrics", {}).get("resumed_species"),
        "taxa_rows": manifest.get("taxa_rows"),
        "name_rows": manifest.get("name_rows"),
        "query_definition_rows": manifest.get("query_definition_rows"),
        "qa_fatal_count": manifest.get("qa_fatal_count"),
        "qa_warning_count": manifest.get("qa_warning_count"),
        "unsupported_metrics": {
            "rss_peak_memory": "not_instrumented",
            "gpu_memory": "not_applicable",
        },
    }


def _write_reports(report: dict[str, Any], *, report_dir: Path, registry_version: str) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"registry_build_{registry_version}"
    json_path = report_dir / f"{stem}.json"
    md_path = report_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_report_markdown(report), encoding="utf-8")
    return {"report_json": str(json_path), "report_md": str(md_path)}


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# Registry Build {report['registry_version']}",
            "",
            f"- Status: {report['status']}",
            f"- Source: {report['source']} {report['source_version']}",
            f"- Taxa rows: {report['taxa_rows']}",
            f"- Name rows: {report['name_rows']}",
            f"- Query definitions: {report['query_definition_rows']}",
            f"- QA fatal: {report['qa_fatal_count']}",
            f"- QA warning: {report['qa_warning_count']}",
            "",
        ]
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "not_instrumented"
