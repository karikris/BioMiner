from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from biominer.registry.compiler import compile_registry_fixture, compile_registry_frames
from biominer.registry.enrichment import DEFAULT_ENRICHMENT_SOURCES, INATURALIST_DAILY_REQUEST_LIMIT, build_enrichment_sources_from_registry, compile_enriched_registry
from biominer.registry.gbif_production import ProductionGBIFClient
from biominer.registry.gbif_source import build_gbif_source_snapshot
from biominer.registry.scope import load_scope
from biominer.registry.translation_harvester import build_translation_candidates_from_registry
from biominer.registry.translation_sources import DEFAULT_TRANSLATION_SOURCES, DEFAULT_TRANSLATION_TARGET_LOCALES_JSON
from biominer.storage.cloud import CloudStorage
from biominer.storage.paths import build_registry_current_pointer, build_registry_current_uri, build_registry_version_uri, safe_path_component
from biominer.storage.uri import is_cloud_uri, join_uri


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
    enrichment_sources: tuple[str, ...] = DEFAULT_ENRICHMENT_SOURCES,
    translation_sources: tuple[str, ...] = DEFAULT_TRANSLATION_SOURCES,
    translation_target_locales_json: str | Path = DEFAULT_TRANSLATION_TARGET_LOCALES_JSON,
    skip_translations: bool = False,
    translation_daily_request_limit: int = 10000,
    max_translation_candidates_per_name: int = 0,
    mymemory_email: str | None = None,
    mymemory_key: str | None = None,
    mymemory_allow_machine_translation: bool = False,
    inaturalist_daily_request_limit: int = INATURALIST_DAILY_REQUEST_LIMIT,
    skip_enrichment: bool = False,
    storage: CloudStorage | None = None,
) -> dict[str, Any]:
    options = {
        "output_dir": output_dir,
        "registry_version": registry_version,
        "scope_path": scope_path,
        "source_json": source_json,
        "reuse_source_json": reuse_source_json,
        "report_dir": report_dir,
        "retrieved_at": retrieved_at,
        "workers": workers,
        "progress_every": progress_every,
        "checkpoint_every": checkpoint_every,
        "max_retries": max_retries,
        "enrichment_sources": enrichment_sources,
        "translation_sources": translation_sources,
        "translation_target_locales_json": translation_target_locales_json,
        "skip_translations": skip_translations,
        "translation_daily_request_limit": translation_daily_request_limit,
        "max_translation_candidates_per_name": max_translation_candidates_per_name,
        "mymemory_email": mymemory_email,
        "mymemory_key": mymemory_key,
        "mymemory_allow_machine_translation": mymemory_allow_machine_translation,
        "inaturalist_daily_request_limit": inaturalist_daily_request_limit,
        "skip_enrichment": skip_enrichment,
    }
    if is_cloud_uri(str(output_dir)):
        if storage is None:
            raise ValueError("storage_backend_required_for_cloud_registry")
        return build_cloud_registry(storage=storage, **options)
    return build_local_registry(**options)


def build_cloud_registry(
    *,
    output_dir: str | Path,
    registry_version: str,
    storage: CloudStorage,
    scope_path: str | Path = "config/butterfly_scope.json",
    source_json: str | Path | None = None,
    reuse_source_json: bool = False,
    report_dir: str | Path = "reports",
    retrieved_at: str | None = None,
    workers: int = 8,
    progress_every: int = 100,
    checkpoint_every: int = 500,
    max_retries: int = 5,
    enrichment_sources: tuple[str, ...] = DEFAULT_ENRICHMENT_SOURCES,
    translation_sources: tuple[str, ...] = DEFAULT_TRANSLATION_SOURCES,
    translation_target_locales_json: str | Path = DEFAULT_TRANSLATION_TARGET_LOCALES_JSON,
    skip_translations: bool = False,
    translation_daily_request_limit: int = 10000,
    max_translation_candidates_per_name: int = 0,
    mymemory_email: str | None = None,
    mymemory_key: str | None = None,
    mymemory_allow_machine_translation: bool = False,
    inaturalist_daily_request_limit: int = INATURALIST_DAILY_REQUEST_LIMIT,
    skip_enrichment: bool = False,
) -> dict[str, Any]:
    if not skip_enrichment:
        raise NotImplementedError("cloud_registry_enrichment_not_implemented")
    base_prefix = str(output_dir).rstrip("/")
    registry_prefix = join_uri(base_prefix, "registry", f"version={safe_path_component(registry_version)}")
    source_uri = build_registry_version_uri(base_prefix, registry_version=registry_version, filename="gbif_source_snapshot.json")
    retrieved = retrieved_at or datetime.now(UTC).isoformat()
    source_from_storage = False
    if reuse_source_json:
        if source_json is None:
            raise FileNotFoundError("--reuse-source-json requires --source-json for cloud registry builds")
        source_path = Path(source_json)
        if not source_path.exists():
            raise FileNotFoundError(f"--reuse-source-json requires an existing source JSON: {source_path}")
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    elif storage.exists(source_uri):
        logger.info("registry.build.resume.source_snapshot source=%s", source_uri)
        source_payload = storage.read_json(source_uri)
        source_from_storage = True
    else:
        with ProductionGBIFClient(max_retries=max_retries, max_connections=workers) as client:
            source_payload = build_gbif_source_snapshot(
                client,
                load_scope(scope_path),
                retrieved_at=retrieved,
                checkpoint_dir=None,
                workers=workers,
                progress_every=progress_every,
                checkpoint_every=checkpoint_every,
                max_retries=max_retries,
                client_factory=lambda: ProductionGBIFClient(max_retries=max_retries, max_connections=workers),
            )
    if not source_from_storage:
        storage.write_json(source_uri, source_payload)
    frames, manifest = compile_registry_frames(
        source_payload,
        source_ref=source_uri,
        output_ref=registry_prefix,
        registry_version=registry_version,
        scope_path=scope_path,
    )
    for filename, frame in frames.items():
        storage.write_parquet_shard(
            build_registry_version_uri(base_prefix, registry_version=registry_version, filename=filename),
            frame,
        )
    manifest_uri = build_registry_version_uri(base_prefix, registry_version=registry_version, filename="manifest.json")
    storage.write_json(manifest_uri, manifest)
    current_pointer_uri = None
    if manifest.get("qa_status") == "passed":
        current_pointer_uri = build_registry_current_uri(base_prefix, filename="manifest.json")
        storage.write_json(
            current_pointer_uri,
            build_registry_current_pointer(
                registry_version=registry_version,
                registry_prefix=registry_prefix,
                manifest_uri=manifest_uri,
                promoted_at=datetime.now(UTC).isoformat(),
            ),
        )
    report = _build_report(
        manifest=manifest,
        source_payload=source_payload,
        source_path=source_uri,
        output_dir=registry_prefix,
        registry_version=registry_version,
        retrieved_at=retrieved,
        enrichment_sources=enrichment_sources,
        translation_sources=(),
        inaturalist_daily_request_limit=inaturalist_daily_request_limit,
        skip_enrichment=skip_enrichment,
        enrichment_manifest=None,
    )
    report_paths = _write_cloud_reports(
        report,
        storage=storage,
        report_dir=str(report_dir),
        output_dir=base_prefix,
        registry_version=registry_version,
    )
    return {
        "registry_version": registry_version,
        "source_json": source_uri,
        "output_dir": registry_prefix,
        "registry_prefix": registry_prefix,
        "manifest_uri": manifest_uri,
        "current_pointer_uri": current_pointer_uri,
        "manifest": manifest,
        **report_paths,
    }


def build_local_registry(
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
    enrichment_sources: tuple[str, ...] = DEFAULT_ENRICHMENT_SOURCES,
    translation_sources: tuple[str, ...] = DEFAULT_TRANSLATION_SOURCES,
    translation_target_locales_json: str | Path = DEFAULT_TRANSLATION_TARGET_LOCALES_JSON,
    skip_translations: bool = False,
    translation_daily_request_limit: int = 10000,
    max_translation_candidates_per_name: int = 0,
    mymemory_email: str | None = None,
    mymemory_key: str | None = None,
    mymemory_allow_machine_translation: bool = False,
    inaturalist_daily_request_limit: int = INATURALIST_DAILY_REQUEST_LIMIT,
    skip_enrichment: bool = False,
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

    base_dir = output / "checkpoints" / "base_registry"
    logger.info("registry.build.compile_base.start source=%s base_dir=%s", source_path, base_dir)
    manifest = compile_registry_fixture(
        source_path,
        base_dir,
        registry_version=registry_version,
        scope_path=scope_path,
    )
    logger.info(
        "registry.build.compile_base.complete status=%s taxa=%s names=%s queries=%s",
        manifest.get("qa_status"),
        manifest.get("taxa_rows"),
        manifest.get("name_rows"),
        manifest.get("query_definition_rows"),
    )
    enrichment_manifest: dict[str, Any] | None = None
    if skip_enrichment:
        logger.info("registry.build.enrichment.skip output=%s", output)
        manifest = compile_registry_fixture(
            source_path,
            output,
            registry_version=registry_version,
            scope_path=scope_path,
        )
    else:
        run_id = _run_id(retrieved)
        enrichment_dir = output / "checkpoints" / "enrichment" / run_id
        logger.info(
            "registry.build.enrichment.start base_dir=%s enrichment_dir=%s sources=%s",
            base_dir,
            enrichment_dir,
            ",".join(enrichment_sources),
        )
        enrichment_manifest = build_enrichment_sources_from_registry(
            registry_dir=base_dir,
            enrichment_dir=enrichment_dir,
            sources=enrichment_sources,
            workers=workers,
            progress_every=progress_every,
            checkpoint_every=checkpoint_every,
            max_retries=max_retries,
            inaturalist_daily_request_limit=inaturalist_daily_request_limit,
            report_dir=report_dir,
        )
        effective_translation_sources = () if skip_translations else tuple(source for source in translation_sources if source)
        if effective_translation_sources:
            logger.info(
                "registry.build.translation.start base_dir=%s enrichment_dir=%s sources=%s",
                base_dir,
                enrichment_dir,
                ",".join(effective_translation_sources),
            )
            translation_manifest = build_translation_candidates_from_registry(
                registry_dir=base_dir,
                enrichment_dir=enrichment_dir,
                translation_sources=effective_translation_sources,
                target_locales_json=translation_target_locales_json,
                max_retries=max_retries,
                daily_request_limit=translation_daily_request_limit,
                max_candidates_per_name=max_translation_candidates_per_name,
                mymemory_email=mymemory_email,
                mymemory_key=mymemory_key,
                mymemory_allow_machine_translation=mymemory_allow_machine_translation,
            )
            enrichment_manifest = {**enrichment_manifest, **translation_manifest}
            logger.info(
                "registry.build.translation.complete status=%s wikimedia_assertions=%s mymemory_candidates=%s requests=%s",
                translation_manifest.get("translation_status"),
                translation_manifest.get("wikimedia_assertion_rows"),
                translation_manifest.get("mymemory_candidate_rows"),
                translation_manifest.get("translation_request_rows"),
            )
        else:
            logger.info("registry.build.translation.skip output=%s", enrichment_dir)
        logger.info(
            "registry.build.compile_enriched.start base_dir=%s enrichment_dir=%s output=%s",
            base_dir,
            enrichment_dir,
            output / "checkpoints" / "canonical" / run_id,
        )
        canonical_dir = output / "checkpoints" / "canonical" / run_id
        manifest = compile_enriched_registry(
            base_registry_dir=base_dir,
            enrichment_dir=enrichment_dir,
            output_dir=canonical_dir,
            registry_version=registry_version,
            scope_path=scope_path,
            requested_sources=enrichment_sources,
            requested_translation_sources=effective_translation_sources,
        )
        logger.info(
            "registry.build.compile_enriched.complete status=%s taxa=%s names=%s queries=%s enrichment_names=%s source_errors=%s",
            manifest.get("qa_status"),
            manifest.get("taxa_rows"),
            manifest.get("name_rows"),
            manifest.get("query_definition_rows"),
            manifest.get("enabled_enrichment_name_rows"),
            manifest.get("source_error_rows"),
        )
        if manifest.get("qa_status") == "passed":
            manifest = {**manifest, "output_dir": str(output), "registry_dir": str(output)}
            _promote_canonical_registry(canonical_dir, output, manifest=manifest)
            logger.info("registry.build.promote.complete canonical_dir=%s output=%s", canonical_dir, output)
        else:
            logger.info("registry.build.promote.blocked status=%s canonical_dir=%s output=%s", manifest.get("qa_status"), canonical_dir, output)
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    report = _build_report(
        manifest=manifest,
        source_payload=source_payload,
        source_path=source_path,
        output_dir=output,
        registry_version=registry_version,
        retrieved_at=retrieved,
        enrichment_sources=enrichment_sources,
        translation_sources=effective_translation_sources if not skip_enrichment else (),
        inaturalist_daily_request_limit=inaturalist_daily_request_limit,
        skip_enrichment=skip_enrichment,
        enrichment_manifest=enrichment_manifest,
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
    enrichment_sources: tuple[str, ...],
    translation_sources: tuple[str, ...],
    inaturalist_daily_request_limit: int,
    skip_enrichment: bool,
    enrichment_manifest: dict[str, Any] | None,
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
        "enrichment_enabled": not skip_enrichment,
        "enrichment_sources": list(enrichment_sources) if not skip_enrichment else [],
        "translation_sources": list(translation_sources) if not skip_enrichment else [],
        "translation_target_locale_count": (enrichment_manifest or {}).get("translation_target_locale_count"),
        "translation_request_rows": (enrichment_manifest or {}).get("translation_request_rows"),
        "wikimedia_assertion_rows": (enrichment_manifest or {}).get("wikimedia_assertion_rows"),
        "mymemory_candidate_rows": (enrichment_manifest or {}).get("mymemory_candidate_rows"),
        "translation_error_rows": (enrichment_manifest or {}).get("translation_error_rows"),
        "inaturalist_daily_request_limit": inaturalist_daily_request_limit if not skip_enrichment else None,
        "enrichment_name_assertion_rows": manifest.get("enrichment_name_assertion_rows"),
        "enabled_enrichment_name_rows": manifest.get("enabled_enrichment_name_rows"),
        "enabled_t5_name_rows": manifest.get("enabled_t5_name_rows"),
        "t5_query_definition_rows": manifest.get("t5_query_definition_rows"),
        "name_candidate_rows": manifest.get("name_candidate_rows"),
        "external_taxon_link_rows": manifest.get("external_taxon_link_rows"),
        "source_error_rows": manifest.get("source_error_rows"),
        "enrichment_status": (enrichment_manifest or {}).get("status"),
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


def _write_cloud_reports(
    report: dict[str, Any],
    *,
    storage: CloudStorage,
    report_dir: str,
    output_dir: str,
    registry_version: str,
) -> dict[str, str]:
    report_base = report_dir.rstrip("/") if is_cloud_uri(report_dir) else join_uri(output_dir, "reports")
    stem = f"registry_build_{safe_path_component(registry_version)}"
    json_uri = join_uri(report_base, f"{stem}.json")
    storage.write_json(json_uri, report)
    return {"report_json": json_uri}


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
            f"- Enrichment enabled: {report['enrichment_enabled']}",
            f"- Enrichment sources: {', '.join(report['enrichment_sources'])}",
            f"- Translation sources: {', '.join(report['translation_sources'])}",
            f"- Translation target locales: {report['translation_target_locale_count']}",
            f"- Translation requests: {report['translation_request_rows']}",
            f"- Wikimedia assertion rows: {report['wikimedia_assertion_rows']}",
            f"- MyMemory candidate rows: {report['mymemory_candidate_rows']}",
            f"- iNaturalist daily request limit: {report['inaturalist_daily_request_limit']}",
            f"- Enabled enrichment names: {report['enabled_enrichment_name_rows']}",
            f"- Enabled T5 names: {report['enabled_t5_name_rows']}",
            f"- T5 query definitions: {report['t5_query_definition_rows']}",
            f"- Source errors: {report['source_error_rows']}",
            "",
        ]
    )


def _run_id(retrieved_at: str) -> str:
    return retrieved_at.replace(":", "").replace("-", "").replace(".", "").replace("+", "Z")


def _promote_canonical_registry(staged_dir: Path, output_dir: Path, *, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name in _canonical_registry_files():
        source = staged_dir / file_name
        if not source.exists():
            continue
        tmp = output_dir / f".{file_name}.tmp"
        shutil.copy2(source, tmp)
        os.replace(tmp, output_dir / file_name)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _canonical_registry_files() -> tuple[str, ...]:
    return (
        "taxa.parquet",
        "taxon_relations.parquet",
        "names.parquet",
        "name_evidence.parquet",
        "source_snapshots.parquet",
        "flickr_query_definitions.parquet",
        "qa_findings.parquet",
        "source_name_assertions.parquet",
        "external_taxon_links.parquet",
        "source_error_records.parquet",
        "source_work_ledger.parquet",
        "translation_candidates.parquet",
        "translation_work_ledger.parquet",
        "name_candidates.parquet",
        "combined_source_snapshot.json",
        "enrichment_coverage.json",
        "enrichment_coverage.md",
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "not_instrumented"
