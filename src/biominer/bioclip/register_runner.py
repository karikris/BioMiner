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
from biominer.bioclip.candidate_sets import (
    CandidateMode,
    CandidateSet,
    CandidateStrategy,
    build_candidate_set,
)
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
        return_image_embeddings: bool = False,
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
    candidate_set_count: int
    avg_records_per_candidate_set: float
    max_records_per_candidate_set: int
    text_embedding_cache_hit_proxy: int
    embedding_output_path: Path | None = None
    embeddings_written: int = 0


@dataclass(frozen=True)
class _RegisterItem:
    record: dict[str, Any]
    base: dict[str, object]
    cached: CachedImage


@dataclass(frozen=True)
class _CandidateGroup:
    candidate_set: CandidateSet
    items: list[_RegisterItem]


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
    classification_mode: str | CandidateMode = CandidateMode.HYBRID,
    candidate_strategy: str | CandidateStrategy = CandidateStrategy.ALL,
    candidate_limit: int | None = None,
    geo_species_index: pl.DataFrame | None = None,
    geo_grid_level: str = "G4_5deg",
    geo_min_species_per_cell: int = 5,
    geo_include_neighbours: bool = False,
    target_species: str | None = None,
    emit_image_embeddings: bool = False,
    embedding_output: str | Path | None = None,
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
    candidate_set_record_counts: dict[str, int] = {}
    embedding_rows: list[dict[str, object]] = []
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
                processed, failed, cleaned, group_counts = _classify_register(
                    register_id=register_id,
                    items=fill.items,
                    classifier=classifier,
                    label_sets=label_sets,
                    species_prompt_variants=species_variants,
                    species_by_label=species_by_label,
                    taxon_metadata_by_name=taxon_metadata_by_name,
                    species_candidates=species_candidates,
                    classification_mode=classification_mode,
                    candidate_strategy=candidate_strategy,
                    candidate_limit=candidate_limit,
                    geo_species_index=geo_species_index,
                    geo_grid_level=geo_grid_level,
                    geo_min_species_per_cell=geo_min_species_per_cell,
                    geo_include_neighbours=geo_include_neighbours,
                    target_species=target_species,
                    emit_image_embeddings=emit_image_embeddings,
                    rows=rows,
                    embedding_rows=embedding_rows,
                    cache_root=Path(cache_root),
                    processed_keys=processed_keys,
                )
                classified += processed
                bioclip_failures += failed
                deleted += cleaned
                for signature, count in group_counts.items():
                    candidate_set_record_counts[signature] = candidate_set_record_counts.get(signature, 0) + count
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
    combined = _stable_sort_frame(combined)
    write_parquet(combined, output)
    write_bucket_views(combined, bucket_views_dir or output.parent)
    embedding_output_path = Path(embedding_output) if embedding_output else None
    embeddings_written = 0
    if emit_image_embeddings and embedding_output_path is not None:
        embeddings_written = _write_embedding_rows(embedding_rows, embedding_output_path)
    candidate_set_count = len(candidate_set_record_counts)
    classified_for_candidate_sets = sum(candidate_set_record_counts.values())
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
        candidate_set_count=candidate_set_count,
        avg_records_per_candidate_set=(
            classified_for_candidate_sets / candidate_set_count if candidate_set_count else 0.0
        ),
        max_records_per_candidate_set=max(candidate_set_record_counts.values(), default=0),
        text_embedding_cache_hit_proxy=sum(max(0, count - 1) for count in candidate_set_record_counts.values()),
        embedding_output_path=embedding_output_path,
        embeddings_written=embeddings_written,
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
        "candidate_set_count": result.candidate_set_count,
        "avg_records_per_candidate_set": result.avg_records_per_candidate_set,
        "max_records_per_candidate_set": result.max_records_per_candidate_set,
        "text_embedding_cache_hit_proxy": result.text_embedding_cache_hit_proxy,
        "embedding_output_path": str(result.embedding_output_path) if result.embedding_output_path else None,
        "embeddings_written": result.embeddings_written,
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
    species_candidates: Sequence[SpeciesCandidate],
    classification_mode: str | CandidateMode,
    candidate_strategy: str | CandidateStrategy,
    candidate_limit: int | None,
    geo_species_index: pl.DataFrame | None,
    geo_grid_level: str,
    geo_min_species_per_cell: int,
    geo_include_neighbours: bool,
    target_species: str | None,
    emit_image_embeddings: bool,
    rows: list[dict[str, object]],
    embedding_rows: list[dict[str, object]],
    cache_root: Path,
    processed_keys: set[tuple[object, ...]],
) -> tuple[int, int, int, dict[str, int]]:
    if not items:
        return 0, 0, 0, {}
    processed = 0
    failed = 0
    deleted = 0
    groups = _group_items_by_candidate_set(
        items,
        species_candidates=species_candidates,
        classification_mode=classification_mode,
        candidate_strategy=candidate_strategy,
        candidate_limit=candidate_limit,
        geo_species_index=geo_species_index,
        geo_grid_level=geo_grid_level,
        geo_min_species_per_cell=geo_min_species_per_cell,
        geo_include_neighbours=geo_include_neighbours,
    )
    group_counts = {group.candidate_set.signature: len(group.items) for group in groups}
    for group in groups:
        active_label_sets: dict[str, Sequence[str]] = {
            name: tuple(labels)
            for name, labels in group.candidate_set.label_sets.items()
        }
        active_species_variants: Sequence[PromptVariant] = group.candidate_set.species_prompt_variants
        if not active_label_sets:
            active_label_sets = label_sets
            active_species_variants = species_prompt_variants
        images = [_image_payload(item, group.candidate_set, target_species=target_species) for item in group.items]
        try:
            predictions = _classify_images(
                classifier,
                images,
                label_sets=active_label_sets,
                species_prompt_variants=active_species_variants,
                return_image_embeddings=emit_image_embeddings,
                top_k=20,
            )
            if len(predictions) != len(group.items):
                raise RuntimeError(f"BioCLIP returned {len(predictions)} predictions for {len(group.items)} images")
            predictions = _rerank_species_top20(
                classifier=classifier,
                items=group.items,
                predictions=predictions,
                candidate_set=group.candidate_set,
                target_species=target_species,
                species_by_label=species_by_label,
            )
            for item, prediction in zip(group.items, predictions, strict=True):
                rows.append(_success_row(item, prediction, register_id, species_by_label, taxon_metadata_by_name, cache_root, group.candidate_set))
                if emit_image_embeddings:
                    embedding_row = _embedding_row(item, prediction)
                    if embedding_row is not None:
                        embedding_rows.append(embedding_row)
                if rows[-1]["image_deleted_after_classification"]:
                    deleted += 1
                processed_keys.add(_dedupe_key(item.base))
                processed += 1
        except Exception as exc:  # noqa: BLE001 - isolate bad images through the same persistent classifier.
            for item in group.items:
                image = _image_payload(item, group.candidate_set, target_species=target_species)
                try:
                    prediction = _classify_images(
                        classifier,
                        [image],
                        label_sets=active_label_sets,
                        species_prompt_variants=active_species_variants,
                        return_image_embeddings=emit_image_embeddings,
                        top_k=20,
                    )[0]
                    prediction = _rerank_species_top20(
                        classifier=classifier,
                        items=[item],
                        predictions=[prediction],
                        candidate_set=group.candidate_set,
                        target_species=target_species,
                        species_by_label=species_by_label,
                    )[0]
                    rows.append(_success_row(item, prediction, register_id, species_by_label, taxon_metadata_by_name, cache_root, group.candidate_set))
                    if emit_image_embeddings:
                        embedding_row = _embedding_row(item, prediction)
                        if embedding_row is not None:
                            embedding_rows.append(embedding_row)
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
                        | {
                            "register_id": register_id,
                            "candidate_set_signature": group.candidate_set.signature,
                            "candidate_set_label_count": group.candidate_set.label_count,
                            "classification_mode": group.candidate_set.mode.value,
                            "candidate_strategy": group.candidate_set.strategy.value,
                        }
                    )
                    failed += 1
                    if image_deleted:
                        deleted += 1
    return processed, failed, deleted, group_counts


def _classify_images(
    classifier: RegisterBatchClassifier,
    images: Sequence[dict[str, object]],
    *,
    label_sets: dict[str, Sequence[str]],
    species_prompt_variants: Sequence[PromptVariant],
    return_image_embeddings: bool,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    if return_image_embeddings:
        return classifier.classify_images_with_label_sets(
            images,
            label_sets=label_sets,
            species_prompt_variants=species_prompt_variants,
            top_k=top_k,
            return_image_embeddings=True,
        )
    return classifier.classify_images_with_label_sets(
        images,
        label_sets=label_sets,
        species_prompt_variants=species_prompt_variants,
        top_k=top_k,
    )


def _rerank_species_top20(
    *,
    classifier: RegisterBatchClassifier,
    items: Sequence[_RegisterItem],
    predictions: Sequence[dict[str, Any]],
    candidate_set: CandidateSet,
    target_species: str | None,
    species_by_label: dict[str, str],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any] | None] = [None] * len(predictions)
    grouped: dict[tuple[str, ...], list[int]] = {}
    candidates_by_key: dict[tuple[str, ...], list[SpeciesCandidate]] = {}
    for index, prediction in enumerate(predictions):
        top20_candidates = _top20_candidates_from_prediction(prediction, candidate_set.species_candidates, species_by_label)
        if not top20_candidates:
            enriched[index] = _with_species_result_columns(prediction, None, candidate_set)
            continue
        key = tuple(candidate.scientific_name for candidate in top20_candidates)
        grouped.setdefault(key, []).append(index)
        candidates_by_key[key] = top20_candidates
    for key, indexes in grouped.items():
        top20_candidates = candidates_by_key[key]
        variants = tuple(species_prompt_variants(list(top20_candidates)))
        labels = tuple(variant.label for variant in variants)
        if not labels:
            for index in indexes:
                enriched[index] = _with_species_result_columns(predictions[index], None, candidate_set)
            continue
        rerank_predictions = _classify_images(
            classifier,
            [_image_payload(items[index], candidate_set, target_species=target_species) for index in indexes],
            label_sets={"species": labels},
            species_prompt_variants=variants,
            return_image_embeddings=False,
            top_k=5,
        )
        for index, rerank_prediction in zip(indexes, rerank_predictions, strict=True):
            rerank_prediction = {
                **rerank_prediction,
                "species_top1_scientific_name": rerank_prediction.get("species_top1_scientific_name")
                or species_by_label.get(str(rerank_prediction.get("species_top1_label") or "")),
            }
            enriched[index] = _with_species_result_columns(predictions[index], rerank_prediction, candidate_set)
    return [row if row is not None else _with_species_result_columns(prediction, None, candidate_set) for row, prediction in zip(enriched, predictions, strict=True)]


def _top20_candidates_from_prediction(
    prediction: dict[str, Any],
    candidates: Sequence[SpeciesCandidate],
    species_by_label: dict[str, str],
) -> list[SpeciesCandidate]:
    by_name = {candidate.scientific_name: candidate for candidate in candidates}
    ordered_names: list[str] = []
    prompt_topk = prediction.get("species_prompt_topk_json")
    if isinstance(prompt_topk, list):
        for row in prompt_topk[:20]:
            if isinstance(row, dict) and row.get("taxon_key"):
                ordered_names.append(str(row["taxon_key"]))
    species_topk = prediction.get("species_topk_json")
    if isinstance(species_topk, list):
        for row in species_topk[:20]:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or "")
            name = species_by_label.get(label)
            if name:
                ordered_names.append(name)
    selected: list[SpeciesCandidate] = []
    seen: set[str] = set()
    for name in ordered_names:
        candidate = by_name.get(name)
        if candidate is not None and candidate.scientific_name not in seen:
            selected.append(candidate)
            seen.add(candidate.scientific_name)
    return selected[:20]


def _with_species_result_columns(
    prediction: dict[str, Any],
    rerank_prediction: dict[str, Any] | None,
    candidate_set: CandidateSet,
) -> dict[str, Any]:
    species_top20 = _topk_rows(prediction.get("species_prompt_topk_json")) or _topk_rows(prediction.get("species_topk_json"))
    species_top20 = species_top20[:20]
    output = {
        **prediction,
        "species_top20_json": species_top20,
        "species_candidate_count": len(candidate_set.species_candidates),
        "species_candidate_sources_json": candidate_set.species_candidate_sources_json,
        "geo_candidate_cell_id": candidate_set.geo_candidate_cell_id,
        "geo_candidate_grid_level": candidate_set.geo_candidate_grid_level,
        "geo_candidate_fallback_level": candidate_set.geo_candidate_fallback_level,
        "species_entropy": prediction.get("species_entropy", prediction.get("species_topk_entropy")),
        "genus_candidates_by_family_json": json.dumps(
            candidate_set.genus_candidates_by_family or {},
            ensure_ascii=True,
            sort_keys=True,
        ),
    }
    if rerank_prediction is None:
        output.update(
            {
                "species_top5_rerun_json": species_top20[:5],
                "species_final_top1": prediction.get("species_top1_scientific_name"),
                "species_final_top1_score": prediction.get("species_top1_score"),
                "species_final_margin": prediction.get("species_top1_top2_margin"),
                "species_final_entropy": prediction.get("species_entropy", prediction.get("species_topk_entropy")),
                "species_final_evidence_summary_json": _species_evidence_summary(candidate_set, rerun=False),
            }
        )
        return output
    rerank_top5 = _topk_rows(rerank_prediction.get("species_prompt_topk_json")) or _topk_rows(rerank_prediction.get("species_topk_json"))
    rerank_top5 = rerank_top5[:5]
    output.update(
        {
            "species_top5_rerun_json": rerank_top5,
            "species_final_top1": _top1_taxon_name(rerank_prediction) or prediction.get("species_top1_scientific_name"),
            "species_final_top1_score": rerank_prediction.get("species_top1_score", prediction.get("species_top1_score")),
            "species_final_margin": rerank_prediction.get("species_top1_top2_margin", prediction.get("species_top1_top2_margin")),
            "species_final_entropy": rerank_prediction.get("species_entropy", rerank_prediction.get("species_topk_entropy")),
            "species_final_evidence_summary_json": _species_evidence_summary(candidate_set, rerun=True),
        }
    )
    return output


def _topk_rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, dict)]


def _top1_taxon_name(prediction: dict[str, Any]) -> str | None:
    prompt_topk = _topk_rows(prediction.get("species_prompt_topk_json"))
    if prompt_topk and prompt_topk[0].get("taxon_key"):
        return str(prompt_topk[0]["taxon_key"])
    return prediction.get("species_top1_scientific_name")


def _species_evidence_summary(candidate_set: CandidateSet, *, rerun: bool) -> str:
    return json.dumps(
        {
            "candidate_strategy": candidate_set.strategy.value,
            "candidate_mode": candidate_set.mode.value,
            "candidate_count": len(candidate_set.species_candidates),
            "candidate_sources": _json_rows(candidate_set.species_candidate_sources_json),
            "geo_candidate_cell_id": candidate_set.geo_candidate_cell_id,
            "geo_candidate_grid_level": candidate_set.geo_candidate_grid_level,
            "geo_candidate_fallback_level": candidate_set.geo_candidate_fallback_level,
            "pass5_rerun": rerun,
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _json_rows(value: str) -> list[dict[str, object]]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [row for row in decoded if isinstance(row, dict)] if isinstance(decoded, list) else []


def _group_items_by_candidate_set(
    items: Sequence[_RegisterItem],
    *,
    species_candidates: Sequence[SpeciesCandidate],
    classification_mode: str | CandidateMode,
    candidate_strategy: str | CandidateStrategy,
    candidate_limit: int | None,
    geo_species_index: pl.DataFrame | None,
    geo_grid_level: str,
    geo_min_species_per_cell: int,
    geo_include_neighbours: bool,
) -> list[_CandidateGroup]:
    grouped: dict[str, _CandidateGroup] = {}
    for item in items:
        candidate_set = build_candidate_set(
            item.record,
            species_candidates=species_candidates,
            mode=classification_mode,
            strategy=candidate_strategy,
            candidate_limit=candidate_limit,
            geo_species_index=geo_species_index,
            geo_grid_level=geo_grid_level,
            geo_min_species_per_cell=geo_min_species_per_cell,
            geo_include_neighbours=geo_include_neighbours,
        )
        existing = grouped.get(candidate_set.signature)
        if existing is None:
            grouped[candidate_set.signature] = _CandidateGroup(candidate_set=candidate_set, items=[item])
        else:
            existing.items.append(item)
    return list(grouped.values())


def _image_payload(item: _RegisterItem, candidate_set: CandidateSet, *, target_species: str | None) -> dict[str, object]:
    return {
        "flickr_photo_id": str(item.base["flickr_photo_id"]),
        "image_path": item.cached.path,
        "image_hash": item.cached.image_hash,
        "image_url_used": item.cached.source_url,
        "resolved_scientific_name": _resolved_scientific_name(item.record, candidate_set, target_species=target_species),
        "candidate_set_signature": candidate_set.signature,
        "candidate_set_label_count": candidate_set.label_count,
        "classification_mode": candidate_set.mode.value,
        "candidate_strategy": candidate_set.strategy.value,
        "text_evidence_present": bool(
            item.record.get("title") or item.record.get("description") or item.record.get("tags") or item.record.get("machine_tags")
        ),
    }


def _resolved_scientific_name(record: dict[str, Any], candidate_set: CandidateSet, *, target_species: str | None) -> str:
    if target_species:
        return target_species
    for key in (
        "resolved_scientific_name",
        "species_final_top1",
        "species_top1_scientific_name",
        "accepted_scientific_name",
        "scientific_name",
        "scientificName",
        "target_species",
    ):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    if len(candidate_set.species_candidates) == 1:
        return candidate_set.species_candidates[0].scientific_name
    return ""


def _embedding_row(item: _RegisterItem, prediction: dict[str, Any]) -> dict[str, object] | None:
    embedding = prediction.get("image_embedding")
    if not isinstance(embedding, list):
        return None
    vector = [float(value) for value in embedding]
    return {
        "image_hash": item.cached.image_hash,
        "source": item.base.get("source"),
        "source_record_id": item.base.get("source_record_id"),
        "flickr_photo_id": item.base.get("flickr_photo_id"),
        "model_id": item.base.get("model_id"),
        "model_version": item.base.get("model_version"),
        "model_checkpoint": item.base.get("model_checkpoint"),
        "preprocessing_version": str(prediction.get("preprocessing_version") or "open_clip_default"),
        "embedding_dimension": len(vector),
        "embedding": vector,
        "created_at": prediction.get("created_at") or datetime.now(UTC).isoformat(),
    }


def _write_embedding_rows(rows: list[dict[str, object]], output: Path) -> int:
    schema = {
        "image_hash": pl.Utf8,
        "source": pl.Utf8,
        "source_record_id": pl.Utf8,
        "flickr_photo_id": pl.Utf8,
        "model_id": pl.Utf8,
        "model_version": pl.Utf8,
        "model_checkpoint": pl.Utf8,
        "preprocessing_version": pl.Utf8,
        "embedding_dimension": pl.Int64,
        "embedding": pl.List(pl.Float64),
        "created_at": pl.Utf8,
    }
    new_frame = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    if output.exists():
        existing = pl.read_parquet(output)
        combined = pl.concat([existing, new_frame], how="diagonal_relaxed")
    else:
        combined = new_frame
    if combined.height:
        combined = combined.unique(
            subset=["image_hash", "model_checkpoint", "preprocessing_version"],
            keep="last",
            maintain_order=True,
        ).sort(["image_hash", "model_checkpoint", "preprocessing_version"])
    write_parquet(combined, output)
    return len(rows)


def _stable_sort_frame(frame: pl.DataFrame) -> pl.DataFrame:
    sort_columns = [
        column
        for column in ("source_record_id", "flickr_photo_id", "source_record_hash", "classification_status")
        if column in frame.columns
    ]
    return frame.sort(sort_columns) if sort_columns and frame.height else frame


def _success_row(
    item: _RegisterItem,
    prediction: dict[str, Any],
    register_id: str,
    species_by_label: dict[str, str],
    taxon_metadata_by_name: dict[str, dict[str, str | None]],
    cache_root: Path,
    candidate_set: CandidateSet,
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
        "candidate_set_signature": candidate_set.signature,
        "candidate_set_label_count": candidate_set.label_count,
        "classification_mode": candidate_set.mode.value,
        "candidate_strategy": candidate_set.strategy.value,
        **_prediction_fields(enriched_prediction),
        **triage,
        "image_deleted_after_classification": image_deleted,
        "model_load_policy": "one persistent BioCLIP 2.5 worker for the full run",
    }
