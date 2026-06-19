from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Callable, Protocol

import polars as pl

from biominer.bioclip.bioclip import DEFAULT_TRIAGE_LABELS
from biominer.bioclip.async_image_cache import cache_images_async
from biominer.bioclip.image_cache import CachedImage, cache_image_from_url
from biominer.bioclip.prompt_templates import PromptVariant
from biominer.bioclip.species_candidates import (
    SpeciesCandidate,
    label_to_scientific_name,
    species_prompt_variants,
    taxon_metadata_by_scientific_name,
)
from biominer.bioclip.temp_image_store import cleanup_cached_image
from biominer.bioclip.triage import (
    _base_row,
    _dedupe_key,
    _empty_triage_frame,
    _failure_row,
    _prediction_fields,
    _read_existing,
    _successful_keys,
    _timestamp,
    classify_bioclip_triage,
)
from biominer.storage.parquet import write_bucket_views, write_parquet


CacheImage = Callable[..., CachedImage]


class RegisterBatchClassifier(Protocol):
    def classify_images_with_label_sets(
        self,
        images: Sequence[dict[str, object]],
        *,
        label_sets: dict[str, Sequence[str]],
        species_prompt_variants: Sequence[PromptVariant] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class RegisterRunnerResult:
    frame: pl.DataFrame
    output_path: Path
    records_seen: int
    records_classified: int
    records_skipped_existing: int
    download_failures: int
    bioclip_failures: int
    images_deleted_after_classification: int
    max_staged_images: int
    register_count: int
    register_size: int


@dataclass(frozen=True)
class _RegisterItem:
    record: dict[str, Any]
    base: dict[str, object]
    cached: CachedImage


@dataclass(frozen=True)
class _RegisterFill:
    register_id: str
    items: list[_RegisterItem]
    failures: list[dict[str, object]]
    records_seen: int
    skipped_existing: int


def process_records_with_registers(
    records: Iterable[dict[str, Any]],
    *,
    classifier: RegisterBatchClassifier,
    species_candidates: list[SpeciesCandidate],
    output_path: str | Path,
    cache_root: str | Path = "data/cache/images",
    cache_image: CacheImage = cache_image_from_url,
    register_count: int = 4,
    register_size: int = 20,
    download_workers: int = 4,
    source: str = "flickr",
    model_id: str = "bioclip2_5",
    model_version: str = "bioclip2_5_huge",
    model_checkpoint: str = "unknown",
    now: datetime | None = None,
    bucket_views_dir: str | Path | None = None,
) -> RegisterRunnerResult:
    output = Path(output_path)
    existing = _read_existing(output)
    processed_keys = _successful_keys(existing)
    records_iter = iter(records)
    classified_at = _timestamp(now)
    species_variants = species_prompt_variants(species_candidates)
    label_sets: dict[str, Sequence[str]] = {
        "species": [variant.label for variant in species_variants],
        "triage": DEFAULT_TRIAGE_LABELS,
    }
    species_by_label = label_to_scientific_name(species_candidates)
    taxon_metadata_by_name = taxon_metadata_by_scientific_name(species_candidates)

    rows: list[dict[str, object]] = []
    records_seen = 0
    skipped = 0
    classified = 0
    download_failures = 0
    bioclip_failures = 0
    deleted = 0
    max_staged = 0
    staged = 0

    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        pending: dict[Future[_RegisterFill], str] = {}
        for register_index in range(register_count):
            fill = _submit_register_fill(
                pool,
                records_iter,
                register_id=f"register_{register_index}",
                register_size=register_size,
                cache_root=Path(cache_root),
                cache_image=cache_image,
                processed_keys=processed_keys,
                source=source,
                model_id=model_id,
                model_version=model_version,
                model_checkpoint=model_checkpoint,
                classified_at=classified_at,
            )
            if fill is not None:
                pending[fill] = f"register_{register_index}"

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            fills: list[tuple[str, _RegisterFill]] = []
            for future in done:
                register_id = pending.pop(future)
                fill = future.result()
                fills.append((register_id, fill))
                rows.extend(fill.failures)
                records_seen += fill.records_seen
                skipped += fill.skipped_existing
                download_failures += len(fill.failures)
                staged += len(fill.items)
                max_staged = max(max_staged, staged)

            for register_id, fill in fills:
                processed, failed, cleaned = _classify_register(
                    register_id=register_id,
                    items=fill.items,
                    classifier=classifier,
                    label_sets=label_sets,
                    species_prompt_variants=species_variants,
                    species_by_label=species_by_label,
                    taxon_metadata_by_name=taxon_metadata_by_name,
                    rows=rows,
                    cache_root=Path(cache_root),
                    processed_keys=processed_keys,
                )
                classified += processed
                bioclip_failures += failed
                deleted += cleaned
                staged -= len(fill.items)

                refill = _submit_register_fill(
                    pool,
                    records_iter,
                    register_id=register_id,
                    register_size=register_size,
                    cache_root=Path(cache_root),
                    cache_image=cache_image,
                    processed_keys=processed_keys,
                    source=source,
                    model_id=model_id,
                    model_version=model_version,
                    model_checkpoint=model_checkpoint,
                    classified_at=classified_at,
                )
                if refill is not None:
                    pending[refill] = register_id

    new_frame = pl.DataFrame(rows) if rows else _empty_triage_frame()
    combined = pl.concat([existing, new_frame], how="diagonal_relaxed") if existing.height else new_frame
    write_parquet(combined, output)
    write_bucket_views(combined, bucket_views_dir or output.parent)
    return RegisterRunnerResult(
        frame=combined,
        output_path=output,
        records_seen=records_seen,
        records_classified=classified,
        records_skipped_existing=skipped,
        download_failures=download_failures,
        bioclip_failures=bioclip_failures,
        images_deleted_after_classification=deleted,
        max_staged_images=max_staged,
        register_count=register_count,
        register_size=register_size,
    )


def write_register_runner_progress(path: str | Path, result: RegisterRunnerResult) -> None:
    progress = {
        "output_path": str(result.output_path),
        "records_seen": result.records_seen,
        "records_classified": result.records_classified,
        "records_skipped_existing": result.records_skipped_existing,
        "download_failures": result.download_failures,
        "bioclip_failures": result.bioclip_failures,
        "images_deleted_after_classification": result.images_deleted_after_classification,
        "max_staged_images": result.max_staged_images,
        "register_count": result.register_count,
        "register_size": result.register_size,
        "model_load_policy": "one persistent BioCLIP 2.5 worker for the full run",
        "updated_at": datetime.now(UTC).isoformat(),
    }
    Path(path).write_text(json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")


def _submit_register_fill(
    pool: ThreadPoolExecutor,
    records_iter: Any,
    *,
    register_id: str,
    register_size: int,
    cache_root: Path,
    cache_image: CacheImage,
    processed_keys: set[tuple[object, ...]],
    source: str,
    model_id: str,
    model_version: str,
    model_checkpoint: str,
    classified_at: str,
) -> Future[_RegisterFill] | None:
    batch: list[dict[str, Any]] = []
    for _ in range(register_size):
        try:
            batch.append(next(records_iter))
        except StopIteration:
            break
    if not batch:
        return None
    return pool.submit(
        _fill_register,
        register_id,
        batch,
        cache_root,
        cache_image,
        processed_keys,
        source,
        model_id,
        model_version,
        model_checkpoint,
        classified_at,
    )


def _fill_register(
    register_id: str,
    records: list[dict[str, Any]],
    cache_root: Path,
    cache_image: CacheImage,
    processed_keys: set[tuple[object, ...]],
    source: str,
    model_id: str,
    model_version: str,
    model_checkpoint: str,
    classified_at: str,
) -> _RegisterFill:
    items: list[_RegisterItem] = []
    failures: list[dict[str, object]] = []
    skipped_existing = 0
    register_cache_root = cache_root / register_id

    # Phase 1: filter to downloadable records.
    downloadable: list[tuple[dict[str, Any], dict[str, object]]] = []
    for record in records:
        base = _base_row(
            record,
            source=source,
            model_id=model_id,
            model_version=model_version,
            model_checkpoint=model_checkpoint,
            classified_at=classified_at,
        )
        if not base["flickr_photo_id"] or not base["image_url"]:
            failures.append(_failure_row(base, status="invalid_record", error="missing image URL or source record ID", retry_eligible=False))
            continue
        if _dedupe_key(base) in processed_keys:
            skipped_existing += 1
            failures.append(
                {
                    **base,
                    "classification_status": "skipped_existing",
                    "occurrence_bin": "in_review",
                    "bin_reason": "duplicate_successful_record",
                    "triage_bin": "in_review",
                    "triage_reason": "duplicate_successful_record",
                    "register_id": register_id,
                }
            )
            continue
        downloadable.append((record, base))

    if not downloadable:
        return _RegisterFill(register_id=register_id, items=items, failures=failures, records_seen=len(records), skipped_existing=skipped_existing)

    # Phase 2: batch-download using async I/O when using the default
    # downloader; fall back to sequential for injected test fakes.
    if cache_image is cache_image_from_url:
        urls = [str(base["image_url"]) for _, base in downloadable]
        results = cache_images_async(urls, cache_root=str(register_cache_root))
        for (record, base), result in zip(downloadable, results, strict=True):
            if isinstance(result, Exception):
                failures.append(_failure_row(base, status="failed_download", error=str(result), retry_eligible=True))
            else:
                items.append(_RegisterItem(record=record, base=base, cached=result))
    else:
        # Sequential fallback for injected cache_image (tests).
        for record, base in downloadable:
            try:
                cached = cache_image(str(base["image_url"]), cache_root=register_cache_root)
            except Exception as exc:  # noqa: BLE001 - download failures are recorded and processing continues.
                failures.append(_failure_row(base, status="failed_download", error=str(exc), retry_eligible=True))
                continue
            items.append(_RegisterItem(record=record, base=base, cached=cached))

    return _RegisterFill(register_id=register_id, items=items, failures=failures, records_seen=len(records), skipped_existing=skipped_existing)


def _classify_register(
    *,
    register_id: str,
    items: list[_RegisterItem],
    classifier: RegisterBatchClassifier,
    label_sets: dict[str, Sequence[str]],
    species_prompt_variants: Sequence[PromptVariant],
    species_by_label: dict[str, str],
    taxon_metadata_by_name: dict[str, dict[str, str | None]],
    rows: list[dict[str, object]],
    cache_root: Path,
    processed_keys: set[tuple[object, ...]],
) -> tuple[int, int, int]:
    if not items:
        return 0, 0, 0
    processed = 0
    failed = 0
    deleted = 0
    images = [_image_payload(item) for item in items]
    try:
        predictions = classifier.classify_images_with_label_sets(
            images,
            label_sets=label_sets,
            species_prompt_variants=species_prompt_variants,
        )
        if len(predictions) != len(items):
            raise RuntimeError(f"BioCLIP returned {len(predictions)} predictions for {len(items)} images")
        for item, prediction in zip(items, predictions, strict=True):
            rows.append(_success_row(item, prediction, register_id, species_by_label, taxon_metadata_by_name, cache_root))
            if rows[-1]["image_deleted_after_classification"]:
                deleted += 1
            processed_keys.add(_dedupe_key(item.base))
            processed += 1
    except Exception as exc:  # noqa: BLE001 - isolate bad images through the same persistent classifier.
        for item in items:
            try:
                prediction = classifier.classify_images_with_label_sets(
                    [_image_payload(item)],
                    label_sets=label_sets,
                    species_prompt_variants=species_prompt_variants,
                )[0]
                rows.append(_success_row(item, prediction, register_id, species_by_label, taxon_metadata_by_name, cache_root))
                if rows[-1]["image_deleted_after_classification"]:
                    deleted += 1
                processed_keys.add(_dedupe_key(item.base))
                processed += 1
            except Exception as single_exc:  # noqa: BLE001 - record and continue.
                image_deleted = cleanup_cached_image(item.cached, cache_root=cache_root, delete_after_success=True)
                rows.append(
                    _failure_row(
                        item.base,
                        status="failed_bioclip",
                        error=f"{single_exc}; batch_error={exc}",
                        retry_eligible=True,
                        image_hash=item.cached.image_hash,
                        image_downloaded=True,
                        image_deleted_after_classification=image_deleted,
                    )
                    | {"register_id": register_id}
                )
                failed += 1
                if image_deleted:
                    deleted += 1
    return processed, failed, deleted


def _image_payload(item: _RegisterItem) -> dict[str, object]:
    return {
        "flickr_photo_id": str(item.base["flickr_photo_id"]),
        "image_path": item.cached.path,
        "image_hash": item.cached.image_hash,
        "image_url_used": item.cached.source_url,
        "resolved_scientific_name": "Papilio demoleus",
        "text_evidence_present": bool(
            item.record.get("title") or item.record.get("description") or item.record.get("tags") or item.record.get("machine_tags")
        ),
    }


def _success_row(
    item: _RegisterItem,
    prediction: dict[str, Any],
    register_id: str,
    species_by_label: dict[str, str],
    taxon_metadata_by_name: dict[str, dict[str, str | None]],
    cache_root: Path,
) -> dict[str, object]:
    species_name = (
        prediction.get("species_top1_scientific_name")
        or species_by_label.get(str(prediction.get("species_top1_label") or ""))
    )
    taxon_metadata = taxon_metadata_by_name.get(str(species_name or ""), {})
    enriched_prediction = {
        **prediction,
        "species_top1_scientific_name": species_name,
        "species_top1_genus": taxon_metadata.get("genus"),
        "species_top1_family": taxon_metadata.get("family"),
    }
    image_deleted = cleanup_cached_image(item.cached, cache_root=cache_root, delete_after_success=True)
    triage = classify_bioclip_triage(record={**item.record, **item.base}, prediction=enriched_prediction)
    return {
        **item.base,
        "register_id": register_id,
        "image_hash": item.cached.image_hash,
        "image_downloaded": True,
        "classification_status": "success",
        "classification_error": None,
        "retry_eligible": False,
        **_prediction_fields(enriched_prediction),
        **triage,
        "image_deleted_after_classification": image_deleted,
        "model_load_policy": "one persistent BioCLIP 2.5 worker for the full run",
    }
