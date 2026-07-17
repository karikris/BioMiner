from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
import json
from math import sqrt
from pathlib import Path
from time import perf_counter
import tracemalloc
from typing import Any, Mapping, Sequence

import polars as pl

from biominer.bioclip.candidate_sets import build_candidate_set
from biominer.bioclip.classification_modes import (
    DEFAULT_RANK_BEAM_WIDTH,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    ClassificationMode,
    normalize_classification_mode,
)
from biominer.bioclip.object_runner import screen_object_detections
from biominer.bioclip.path_taxonomy_store import PathTaxonomyStore
from biominer.bioclip.taxonomy_embedding_cache import (
    TaxonomyTextEmbeddingIndex,
    build_taxonomy_text_embedding_cache,
)
from biominer.detection.detector_base import DecodedImage, DetectionCandidate
from biominer.detection.pipeline import run_detection_pipeline
from biominer.detection.policy import (
    DetectionPolicy,
    DetectionRunPolicy,
    detection_is_bioclip_eligible,
)
from biominer.evidence.join import write_object_evidence_outputs
from biominer.registry.classification_v3 import (
    CLASSIFICATION_V3_VERSION,
    REVIEWED_RANK_SKIP_EDGE,
    classification_v3_artifact_paths,
    write_classification_v3_artifacts,
)
from biominer.species.context import SpeciesContext
from biominer.storage.parquet import write_parquet
from biominer.vision.rolling_worker import (
    CommitResult,
    DetectionBatch,
    ImageBatch,
    PlannedBatch,
    RollingVisionWorker,
    RollingVisionWorkerSettings,
    ScoreBatch,
    ScoreInputBatch,
)


BENCHMARK_KIND = "vision_plumbing_model_free"
ROLLING_MATRIX_BENCHMARK_KIND = "rolling_vision_worker_model_free_matrix"
BENCHMARK_TAXONOMY_REGISTRY_VERSION = "benchmark-taxonomy-v1"
BENCHMARK_PRIMARY_SPECIES = "Benchmarkus alpha"
BENCHMARK_SECONDARY_SPECIES = "Benchmarkus beta"
BENCHMARK_OUTGROUP_SPECIES = "Metricsus gamma"
BENCHMARK_SECOND_OUTGROUP_SPECIES = "Metricsus delta"
FAKE_IMAGE_WIDTH = 64
FAKE_IMAGE_HEIGHT = 64
FAKE_IMAGE_BYTES = bytes([240, 236, 220]) * FAKE_IMAGE_WIDTH * FAKE_IMAGE_HEIGHT


@dataclass(frozen=True)
class VisionPlumbingBenchmarkResult:
    metrics: dict[str, Any]
    output_dir: Path
    metrics_path: Path
    summary_path: Path


@dataclass(frozen=True)
class RollingWorkerBenchmarkMatrixResult:
    metrics: dict[str, Any]
    output_dir: Path
    metrics_path: Path
    summary_path: Path


class SequentialFakeObjectDetector:
    model_id = "fake-yoloe26-plumbing"
    model_version = "benchmark"
    checkpoint = "model-free"
    backend = "fake"

    def __init__(self, detections_by_record: Sequence[Sequence[DetectionCandidate]]) -> None:
        self._detections_by_record = [tuple(detections) for detections in detections_by_record]
        self._offset = 0

    def detect_batch(self, images: Sequence[DecodedImage]) -> list[list[DetectionCandidate]]:
        start = self._offset
        end = start + len(images)
        self._offset = end
        return [list(detections) for detections in self._detections_by_record[start:end]]


class BenchmarkBioClipScorer:
    model_id = "fake-bioclip"
    model_version = "benchmark"
    model_checkpoint = "model-free"

    def __init__(self) -> None:
        self.score_calls = 0
        self.batch_calls = 0
        self.label_evaluations = 0
        self.image_embedding_calls = 0
        self.images_embedded = 0
        self.scored_detection_ids: list[str] = []
        self.batch_sizes: list[int] = []
        self.image_embedding_batch_sizes: list[int] = []
        self.label_set_names_by_batch: list[list[str]] = []

    def embed_image_items(self, items: Sequence[dict[str, Any]]) -> list[list[float]]:
        self.image_embedding_calls += 1
        self.images_embedded += len(items)
        self.image_embedding_batch_sizes.append(len(items))
        self.scored_detection_ids.extend(str(item.get("detection_id") or "") for item in items)
        return [[1.0, 0.0] for _item in items]

    def score(self, item: dict[str, Any], labels: tuple[str, ...]) -> dict[str, float]:
        self.score_calls += 1
        self.label_evaluations += len(labels)
        self.scored_detection_ids.append(str(item.get("detection_id") or ""))
        return _fake_label_scores(labels)

    def score_label_sets_batch(
        self,
        items: Sequence[dict[str, Any]],
        label_sets: Mapping[str, Sequence[str]],
    ) -> dict[str, list[dict[str, float]]]:
        self.batch_calls += 1
        self.batch_sizes.append(len(items))
        self.label_set_names_by_batch.append(sorted(str(name) for name in label_sets))
        self.scored_detection_ids.extend(str(item.get("detection_id") or "") for item in items)
        self.label_evaluations += len(items) * sum(len(labels) for labels in label_sets.values())
        return {str(name): [_fake_label_scores(tuple(str(label) for label in labels)) for _item in items] for name, labels in label_sets.items()}


def run_vision_plumbing_benchmark(
    *,
    records: int,
    butterfly_rate: float,
    detections_per_butterfly: int,
    classification_mode: ClassificationMode = HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    registry_dir: str | Path | None = None,
    output_dir: str | Path,
    rank_beam_width: int = DEFAULT_RANK_BEAM_WIDTH,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
) -> VisionPlumbingBenchmarkResult:
    record_count = _positive_int(records, name="records", allow_zero=True)
    butterfly_rate_value = _rate(butterfly_rate, name="butterfly_rate")
    detections_per_butterfly_count = _positive_int(detections_per_butterfly, name="detections_per_butterfly")
    mode = normalize_classification_mode(classification_mode)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    taxonomy_path = Path(registry_dir) if registry_dir is not None else output / "taxonomy_store"
    taxonomy_fixture_created = False
    taxonomy_store_reads = 0

    stage_seconds: dict[str, float] = {}
    total_start = perf_counter()
    tracemalloc.start()
    try:
        stage_start = perf_counter()
        canonical = benchmark_canonical_records(record_count, butterfly_rate=butterfly_rate_value)
        detections_by_record = benchmark_detections(
            record_count,
            butterfly_rate=butterfly_rate_value,
            detections_per_butterfly=detections_per_butterfly_count,
        )
        canonical_path = write_parquet(canonical, output / "canonical_source_records.parquet")
        stage_seconds["generate_records"] = _elapsed(stage_start)

        stage_start = perf_counter()
        scorer = BenchmarkBioClipScorer()
        taxonomy_store: PathTaxonomyStore | None = None
        taxonomy_embedding_index: TaxonomyTextEmbeddingIndex | None = None
        if mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION:
            if not taxonomy_path.exists():
                write_benchmark_taxonomy_store(taxonomy_path)
                taxonomy_fixture_created = True
            taxonomy_store = PathTaxonomyStore.read(taxonomy_path)
            taxonomy_store_reads += 1
            taxonomy_embedding_cache = build_taxonomy_text_embedding_cache(
                taxonomy_store,
                model_id=scorer.model_id,
                model_checkpoint=scorer.model_checkpoint,
                embed_labels=_benchmark_text_embeddings,
            )
            taxonomy_embedding_index = TaxonomyTextEmbeddingIndex.from_frame(
                taxonomy_embedding_cache,
                taxonomy_store=taxonomy_store,
                model_id=scorer.model_id,
                model_checkpoint=scorer.model_checkpoint,
            )
        species_context = benchmark_species_context()
        candidate_set = build_candidate_set(
            species_context,
            records=canonical.to_dicts(),
            allow_single_target_fixture=True,
        )
        stage_seconds["load_taxonomy"] = _elapsed(stage_start)

        stage_start = perf_counter()
        detector = SequentialFakeObjectDetector(detections_by_record)
        detection_policy = DetectionPolicy(backend=detector.backend, crop_padding_ratio=0.08, crop_target_px=64)
        run_policy = DetectionRunPolicy(
            download_workers=1,
            decode_workers=1,
            detector_batch_size=16,
            crop_batch_size=32,
            parquet_batch_rows=5000,
        )
        detections_path = output / "object_detections.parquet"
        detection_result = run_detection_pipeline(
            records=canonical.to_dicts(),
            detector=detector,
            output_path=detections_path,
            image_loader=benchmark_image_loader,
            detection_policy=detection_policy,
            run_policy=run_policy,
        )
        stage_seconds["detect_objects"] = _elapsed(stage_start)

        stage_start = perf_counter()
        scores_path = output / "object_bioclip_scores.parquet"
        score_result = screen_object_detections(
            canonical_records=canonical,
            detections=detection_result.frame,
            species_context=species_context,
            candidate_set=candidate_set,
            scorer=scorer,
            output_path=scores_path,
            classification_mode=mode,
            rank_beam_width=rank_beam_width,
            species_first_pass_top_k=species_first_pass_top_k,
            species_rerank_top_k=species_rerank_top_k,
            path_taxonomy_store=(taxonomy_store if mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION else None),
            taxonomy_text_embedding_index=(taxonomy_embedding_index if mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION else None),
            detection_policy=detection_policy,
            parquet_batch_rows=5000,
            bioclip_batch_size=24,
        )
        stage_seconds["score_crops"] = _elapsed(stage_start)

        stage_start = perf_counter()
        evidence_outputs = write_object_evidence_outputs(
            canonical_source_records=canonical,
            object_detections=detection_result.frame,
            object_scores=score_result.frame,
            joined_output_path=output / "object_evidence_joined.parquet",
            photo_summary_output_path=output / "photo_evidence_summary.parquet",
            species_context=species_context,
        )
        joined = pl.read_parquet(evidence_outputs.object_evidence_joined)
        summary = pl.read_parquet(evidence_outputs.photo_evidence_summary)
        stage_seconds["join_evidence"] = _elapsed(stage_start)

        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        metrics_path = output / "benchmark_metrics.json"
        summary_path = output / "benchmark_summary.md"
        metrics = _benchmark_metrics(
            records=record_count,
            butterfly_rate=butterfly_rate_value,
            detections_per_butterfly=detections_per_butterfly_count,
            classification_mode=mode,
            taxonomy_path=taxonomy_path,
            taxonomy_fixture_created=taxonomy_fixture_created,
            taxonomy_store_reads=taxonomy_store_reads,
            temporary_directories_left=_temporary_directories_left(output),
            canonical=canonical,
            detection_result=detection_result,
            score_result=score_result,
            joined=joined,
            summary=summary,
            scorer=scorer,
            stage_seconds=stage_seconds,
            current_tracemalloc_bytes=current_bytes,
            peak_tracemalloc_bytes=peak_bytes,
            total_start=total_start,
            outputs={
                "canonical_source_records": canonical_path,
                "object_detections": detection_result.output_path,
                "object_bioclip_scores": score_result.output_path,
                "object_evidence_joined": evidence_outputs.object_evidence_joined,
                "photo_evidence_summary": evidence_outputs.photo_evidence_summary,
                "benchmark_metrics": metrics_path,
                "benchmark_summary": summary_path,
            },
        )
        report_start = perf_counter()
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        summary_path.write_text(_benchmark_summary_markdown(metrics), encoding="utf-8")
        metrics["elapsed_seconds_by_stage"]["write_reports"] = _elapsed(report_start)
        metrics["elapsed_seconds"] = _elapsed(total_start)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        summary_path.write_text(_benchmark_summary_markdown(metrics), encoding="utf-8")
        return VisionPlumbingBenchmarkResult(
            metrics=metrics,
            output_dir=output,
            metrics_path=metrics_path,
            summary_path=summary_path,
        )
    finally:
        tracemalloc.stop()


def run_rolling_worker_benchmark_matrix(
    *,
    records: int,
    output_dir: str | Path,
) -> RollingWorkerBenchmarkMatrixResult:
    record_count = _positive_int(records, name="records", allow_zero=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    variants = rolling_worker_benchmark_variants()
    rows: list[dict[str, Any]] = []
    total_start = perf_counter()
    source_records = pl.DataFrame(
        {
            "source": ["flickr"] * record_count,
            "flickr_photo_id": [f"rolling-bench-{index:06d}" for index in range(record_count)],
        }
    )
    for index, variant in enumerate(variants):
        worker = RollingVisionWorker(
            settings=RollingVisionWorkerSettings(
                vision_batch_rows=int(variant["vision_batch_rows"]),
                image_prefetch_batches=2,
                accelerator_concurrency=int(variant["accelerator_concurrency"]),
                bioclip_preprocess_workers=int(variant["bioclip_preprocess_workers"]),
            ),
            image_stage=_benchmark_image_stage,
            detection_stage=_benchmark_detection_stage,
            score_input_stage=_benchmark_score_input_stage,
            score_stage=_benchmark_score_stage,
            commit_stage=_benchmark_commit_stage,
        )
        started = perf_counter()
        result = worker.run(source_records)
        elapsed = _elapsed(started)
        rows.append(
            {
                "variant_index": index,
                **variant,
                "records": record_count,
                "elapsed_seconds": elapsed,
                "batches_seen": result.batches_seen,
                "batches_committed": result.batches_committed,
                "staged_images_per_sec": result.metrics.get("staged_images_per_sec"),
                "yolo_images_per_sec": result.metrics.get("yolo_images_per_sec"),
                "bioclip_inputs_per_sec": result.metrics.get("bioclip_inputs_per_sec"),
                "detection_rows_per_image": result.metrics.get("detection_rows_per_image"),
                "bioclip_score_inputs_per_image": result.metrics.get("bioclip_score_inputs_per_image"),
                "max_resident_image_batches": result.metrics.get("max_resident_image_batches"),
                "queue_wait_seconds_by_stage": result.metrics.get("queue_wait_seconds_by_stage"),
                "cleanup_paths_deleted": result.metrics.get("cleanup_paths_deleted"),
                "adaptive_retry_count": result.metrics.get("adaptive_retry_count"),
            }
        )
    metrics_path = output / "rolling_benchmark_matrix_metrics.json"
    summary_path = output / "rolling_benchmark_matrix_summary.md"
    metrics = {
        "benchmark_kind": ROLLING_MATRIX_BENCHMARK_KIND,
        "status": "ok",
        "records": record_count,
        "variant_count": len(rows),
        "dimensions": {
            "yolo_sidecar_transport": ["json_b64", "image_path"],
            "accelerator_concurrency": [1, 2],
            "bioclip_preprocess_workers": [1, 2, 4],
            "vision_batch_rows": [250, 500, 1000],
        },
        "variants": rows,
        "elapsed_seconds": _elapsed(total_start),
        "created_at": datetime.now(UTC).isoformat(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(_rolling_matrix_summary_markdown(metrics), encoding="utf-8")
    return RollingWorkerBenchmarkMatrixResult(
        metrics=metrics,
        output_dir=output,
        metrics_path=metrics_path,
        summary_path=summary_path,
    )


def rolling_worker_benchmark_variants() -> list[dict[str, Any]]:
    return [
        {
            "yolo_sidecar_transport": transport,
            "accelerator_concurrency": accelerator_concurrency,
            "bioclip_preprocess_workers": preprocess_workers,
            "vision_batch_rows": batch_rows,
        }
        for transport, accelerator_concurrency, preprocess_workers, batch_rows in product(
            ("json_b64", "image_path"),
            (1, 2),
            (1, 2, 4),
            (250, 500, 1000),
        )
    ]


def write_benchmark_taxonomy_store(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    write_parquet(benchmark_taxa_frame(), base / "taxa.parquet")
    (base / "manifest.json").write_text(
        json.dumps({"registry_version": BENCHMARK_TAXONOMY_REGISTRY_VERSION}, sort_keys=True),
        encoding="utf-8",
    )
    source_path = base / "benchmark_classification_source.json"
    source_path.write_text(json.dumps(_benchmark_classification_source(), sort_keys=True), encoding="utf-8")
    try:
        write_classification_v3_artifacts(base, source_path=source_path)
    finally:
        source_path.unlink(missing_ok=True)
    return classification_v3_artifact_paths(base)


def benchmark_taxa_frame() -> pl.DataFrame:
    now = datetime.now(UTC).isoformat()
    rows = [
        _taxon_row(
            "gbif:9401",
            BENCHMARK_PRIMARY_SPECIES,
            "gbif:9417",
            "Papilionidae",
            "gbif:9400",
            "Benchmarkus",
            now,
        ),
        _taxon_row(
            "gbif:9402",
            BENCHMARK_SECONDARY_SPECIES,
            "gbif:9417",
            "Papilionidae",
            "gbif:9400",
            "Benchmarkus",
            now,
        ),
        _taxon_row(
            "gbif:7001",
            BENCHMARK_OUTGROUP_SPECIES,
            "gbif:7017",
            "Nymphalidae",
            "gbif:7000",
            "Metricsus",
            now,
        ),
        _taxon_row(
            "gbif:7002",
            BENCHMARK_SECOND_OUTGROUP_SPECIES,
            "gbif:7017",
            "Nymphalidae",
            "gbif:7000",
            "Metricsus",
            now,
        ),
    ]
    return pl.DataFrame(rows)


def _benchmark_classification_source() -> dict[str, Any]:
    reviewed = {
        "reviewed": True,
        "review_status": "reviewed",
        "reviewed_by": "BioMiner benchmark fixture",
        "reviewed_at": "2026-07-11",
        "enabled": True,
    }
    source_id = "benchmark-taxonomy-source"
    lineages = (
        (
            "Papilionidae",
            "Papilioninae",
            "Papilionini",
            "Benchmarkus",
            (
                ("gbif:9401", BENCHMARK_PRIMARY_SPECIES),
                ("gbif:9402", BENCHMARK_SECONDARY_SPECIES),
            ),
        ),
        (
            "Nymphalidae",
            "Nymphalinae",
            "Nymphalini",
            "Metricsus",
            (
                ("gbif:7001", BENCHMARK_OUTGROUP_SPECIES),
                ("gbif:7002", BENCHMARK_SECOND_OUTGROUP_SPECIES),
            ),
        ),
    )
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for family, subfamily, tribe, genus, species_rows in lineages:
        ranked_names = (
            ("FAMILY", family),
            ("SUBFAMILY", subfamily),
            ("TRIBE", tribe),
            ("GENUS", genus),
        )
        parent_id: str | None = None
        for rank, name in ranked_names:
            node_id = f"{rank.casefold()}:{name.casefold()}"
            nodes.append(
                {
                    "node_id": node_id,
                    "rank": rank,
                    "scientific_name": name,
                    "source_id": source_id,
                    "evidence": "model-free benchmark fixture",
                    **reviewed,
                }
            )
            if parent_id is not None:
                edge = {
                    "parent_node_id": parent_id,
                    "child_node_id": node_id,
                    "source_id": source_id,
                    "evidence": "model-free benchmark fixture",
                    **reviewed,
                }
                if rank == "GENUS":
                    edge.update(
                        {
                            "edge_type": REVIEWED_RANK_SKIP_EDGE,
                            "skipped_ranks": ["SUBTRIBE"],
                            "skip_reason": ("reviewed synthetic benchmark fixture has no supported subtribe assertion"),
                        }
                    )
                edges.append(edge)
            parent_id = node_id
        for taxon_key, species in species_rows:
            species_id = f"species:{species.casefold().replace(' ', '-')}"
            nodes.append(
                {
                    "node_id": species_id,
                    "rank": "SPECIES",
                    "scientific_name": species,
                    "source_id": source_id,
                    "evidence": "model-free benchmark fixture",
                    **reviewed,
                }
            )
            edges.append(
                {
                    "parent_node_id": parent_id,
                    "child_node_id": species_id,
                    "source_id": source_id,
                    "evidence": "model-free benchmark fixture",
                    **reviewed,
                }
            )
            mappings.append(
                {
                    "gbif_species_key": taxon_key,
                    "accepted_scientific_name": species,
                    "species_node_id": species_id,
                    "source_id": source_id,
                    "evidence": "exact benchmark key and name",
                    **reviewed,
                }
            )
    return {
        "classification_version": CLASSIFICATION_V3_VERSION,
        "sources": [
            {
                "source_id": source_id,
                "authority": "BioMiner benchmark fixture",
                "release": "v3",
                "citation": "model-free internal benchmark taxonomy",
                "retrieved_at": "2026-07-11",
                "evidence_url": "https://example.invalid/biominer-benchmark-taxonomy",
                "evidence": "Synthetic classification used only by model-free benchmarks.",
            }
        ],
        "nodes": nodes,
        "edges": edges,
        "species_mappings": mappings,
    }


def benchmark_species_context() -> SpeciesContext:
    return SpeciesContext(
        scientific_name=BENCHMARK_PRIMARY_SPECIES,
        accepted_taxon_key="gbif:9401",
        canonical_name=BENCHMARK_PRIMARY_SPECIES,
        family="Papilionidae",
        genus="Benchmarkus",
        family_key="gbif:9417",
        genus_key="gbif:9400",
        species_key="gbif:9401",
        registry_version=BENCHMARK_TAXONOMY_REGISTRY_VERSION,
    )


def benchmark_canonical_records(records: int, *, butterfly_rate: float) -> pl.DataFrame:
    butterfly_records = _butterfly_record_count(records, butterfly_rate=butterfly_rate)
    rows = []
    for index in range(records):
        is_butterfly = index < butterfly_records
        photo_id = f"bench-{index:06d}"
        rows.append(
            {
                "source": "flickr",
                "flickr_photo_id": photo_id,
                "source_record_hash": f"sha256:{photo_id}",
                "image_url": f"memory://benchmark/{photo_id}.jpg",
                "photo_page_url": f"https://example.invalid/photos/{photo_id}",
                "title": f"{BENCHMARK_PRIMARY_SPECIES} benchmark butterfly" if is_butterfly else "benchmark hard negative",
                "description": "model-free generated benchmark record",
                "raw_tags": f"{BENCHMARK_PRIMARY_SPECIES} butterfly" if is_butterfly else "leaf object",
                "tags": f"{BENCHMARK_PRIMARY_SPECIES} butterfly" if is_butterfly else "leaf object",
                "date_taken": "2024-07-01",
                "latitude": -27.4705,
                "longitude": 153.0260,
            }
        )
    return pl.DataFrame(rows)


def benchmark_detections(
    records: int,
    *,
    butterfly_rate: float,
    detections_per_butterfly: int,
) -> list[list[DetectionCandidate]]:
    butterfly_records = _butterfly_record_count(records, butterfly_rate=butterfly_rate)
    detections_by_record: list[list[DetectionCandidate]] = []
    for index in range(records):
        if index < butterfly_records:
            detections_by_record.append(
                [
                    DetectionCandidate(
                        label="butterfly_like",
                        score=0.91,
                        bbox_xyxy=_butterfly_bbox(detection_index),
                        objectness_score=0.95,
                    )
                    for detection_index in range(detections_per_butterfly)
                ]
            )
        else:
            detections_by_record.append(
                [
                    DetectionCandidate(
                        label="hard_negative",
                        score=0.80,
                        bbox_xyxy=(12.0, 12.0, 48.0, 48.0),
                        objectness_score=0.85,
                    )
                ]
            )
    return detections_by_record


def benchmark_image_loader(record: dict[str, Any]) -> DecodedImage:
    return DecodedImage(
        width=FAKE_IMAGE_WIDTH,
        height=FAKE_IMAGE_HEIGHT,
        mode="RGB",
        data=FAKE_IMAGE_BYTES,
        source_uri=str(record.get("image_url") or ""),
    )


def _benchmark_metrics(
    *,
    records: int,
    butterfly_rate: float,
    detections_per_butterfly: int,
    classification_mode: str,
    taxonomy_path: Path,
    taxonomy_fixture_created: bool,
    taxonomy_store_reads: int,
    temporary_directories_left: list[str],
    canonical: pl.DataFrame,
    detection_result: Any,
    score_result: Any,
    joined: pl.DataFrame,
    summary: pl.DataFrame,
    scorer: BenchmarkBioClipScorer,
    stage_seconds: dict[str, float],
    current_tracemalloc_bytes: int,
    peak_tracemalloc_bytes: int,
    total_start: float,
    outputs: Mapping[str, Path | None],
) -> dict[str, Any]:
    detection_rows = detection_result.frame.to_dicts()
    eligible = [row for row in detection_rows if detection_is_bioclip_eligible(row)]
    detected_rows = [row for row in detection_rows if str(row.get("detection_status") or "") == "detected"]
    non_butterfly = [row for row in detected_rows if str(row.get("detector_label") or "") != "butterfly_like"]
    rows = {
        "canonical_source_records": canonical.height,
        "object_detections": detection_result.frame.height,
        "detected_objects": len(detected_rows),
        "object_bioclip_scores": score_result.frame.height,
        "object_evidence_joined": joined.height,
        "photo_evidence_summary": summary.height,
    }
    return {
        "benchmark_kind": BENCHMARK_KIND,
        "status": "ok",
        "records": records,
        "records_generated": canonical.height,
        "butterfly_rate": butterfly_rate,
        "detections_per_butterfly": detections_per_butterfly,
        "classification_mode": classification_mode,
        "registry_dir": str(taxonomy_path),
        "taxonomy_fixture_created": taxonomy_fixture_created,
        "taxonomy_store_reads": int(taxonomy_store_reads),
        "temporary_directories_left": temporary_directories_left,
        "images_loaded": detection_result.images_loaded,
        "image_failures": detection_result.image_failures,
        "detection_rows_written": detection_result.frame.height,
        "detected_object_rows": len(detected_rows),
        "eligible_butterfly_like_detections": len(eligible),
        "butterfly_like_detections": len(eligible),
        "non_butterfly_detections": len(non_butterfly),
        "crops_materialised": detection_result.crops_created,
        "crops_scored": score_result.crops_scored,
        "score_rows_written": score_result.frame.height,
        "joined_evidence_rows": joined.height,
        "photo_summary_rows": summary.height,
        "elapsed_seconds_by_stage": {key: round(value, 6) for key, value in stage_seconds.items()},
        "elapsed_seconds": _elapsed(total_start),
        "current_tracemalloc_bytes": int(current_tracemalloc_bytes),
        "peak_tracemalloc_bytes": int(peak_tracemalloc_bytes),
        "scorer": {
            "score_calls": scorer.score_calls,
            "label_set_batch_calls": scorer.batch_calls,
            "label_evaluations": scorer.label_evaluations,
            "image_embedding_calls": scorer.image_embedding_calls,
            "images_embedded": scorer.images_embedded,
            "scored_detection_ids": list(scorer.scored_detection_ids),
            "batch_sizes": list(scorer.batch_sizes),
            "image_embedding_batch_sizes": list(scorer.image_embedding_batch_sizes),
            "label_set_names_by_batch": list(scorer.label_set_names_by_batch),
        },
        "rows": rows,
        "outputs": {key: str(value) if value is not None else None for key, value in outputs.items()},
    }


def _benchmark_summary_markdown(metrics: Mapping[str, Any]) -> str:
    stage_seconds = dict(metrics.get("elapsed_seconds_by_stage") or {})
    rows = dict(metrics.get("rows") or {})
    lines = [
        "# Vision Plumbing Benchmark",
        "",
        f"- benchmark_kind: `{metrics.get('benchmark_kind')}`",
        f"- status: `{metrics.get('status')}`",
        f"- records: {metrics.get('records')}",
        f"- classification_mode: `{metrics.get('classification_mode')}`",
        f"- images_loaded: {metrics.get('images_loaded')}",
        f"- detection_rows_written: {metrics.get('detection_rows_written')}",
        f"- eligible_butterfly_like_detections: {metrics.get('eligible_butterfly_like_detections')}",
        f"- crops_materialised: {metrics.get('crops_materialised')}",
        f"- crops_scored: {metrics.get('crops_scored')}",
        f"- score_rows_written: {metrics.get('score_rows_written')}",
        f"- joined_evidence_rows: {metrics.get('joined_evidence_rows')}",
        f"- photo_summary_rows: {metrics.get('photo_summary_rows')}",
        f"- peak_tracemalloc_bytes: {metrics.get('peak_tracemalloc_bytes')}",
        f"- elapsed_seconds: {metrics.get('elapsed_seconds')}",
        "",
        "## Stage Seconds",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(stage_seconds.items()))
    lines.extend(["", "## Rows", ""])
    lines.extend(f"- {key}: {value}" for key, value in sorted(rows.items()))
    return "\n".join(lines) + "\n"


def _rolling_matrix_summary_markdown(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Rolling Vision Worker Benchmark Matrix",
        "",
        f"- benchmark_kind: `{metrics.get('benchmark_kind')}`",
        f"- status: `{metrics.get('status')}`",
        f"- records: {metrics.get('records')}",
        f"- variants: {metrics.get('variant_count')}",
        f"- elapsed_seconds: {metrics.get('elapsed_seconds')}",
        "",
        "## Dimensions",
        "",
    ]
    dimensions = dict(metrics.get("dimensions") or {})
    lines.extend(f"- {name}: {values}" for name, values in sorted(dimensions.items()))
    lines.extend(["", "## Variants", ""])
    for row in list(metrics.get("variants") or [])[:10]:
        lines.append(
            "- "
            + ", ".join(
                [
                    f"idx={row.get('variant_index')}",
                    f"transport={row.get('yolo_sidecar_transport')}",
                    f"accel={row.get('accelerator_concurrency')}",
                    f"preprocess={row.get('bioclip_preprocess_workers')}",
                    f"batch={row.get('vision_batch_rows')}",
                    f"elapsed={row.get('elapsed_seconds')}",
                ]
            )
        )
    if int(metrics.get("variant_count") or 0) > 10:
        lines.append(f"- ... {int(metrics.get('variant_count') or 0) - 10} more variants in JSON")
    return "\n".join(lines) + "\n"


def _benchmark_image_stage(planned: PlannedBatch) -> ImageBatch:
    return ImageBatch(
        batch_index=planned.batch_index,
        batch_id=planned.batch_id,
        part_id=planned.part_id,
        records=planned.records,
    )


def _benchmark_detection_stage(batch: ImageBatch) -> DetectionBatch:
    return DetectionBatch(
        image_batch=batch,
        frame=pl.DataFrame(
            {
                "detection_id": [f"{batch.batch_id}-det-{index:06d}" for index in range(batch.records.height)],
            }
        ),
    )


def _benchmark_score_input_stage(batch: DetectionBatch) -> ScoreInputBatch:
    return ScoreInputBatch(
        detection_batch=batch,
        frame=pl.DataFrame({"detection_id": batch.frame.get_column("detection_id").to_list()}),
    )


def _benchmark_score_stage(batch: ScoreInputBatch) -> ScoreBatch:
    return ScoreBatch(
        score_input_batch=batch,
        frame=pl.DataFrame({"detection_id": batch.frame.get_column("detection_id").to_list()}),
    )


def _benchmark_commit_stage(batch: ScoreBatch) -> CommitResult:
    return CommitResult(
        batch_id=batch.score_input_batch.detection_batch.image_batch.batch_id,
        part_outputs={},
        cleanup_paths_deleted=0,
    )


def _fake_label_scores(labels: Sequence[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label in labels:
        text = label.casefold()
        if BENCHMARK_PRIMARY_SPECIES.casefold() in text:
            score = 0.86
        elif BENCHMARK_SECONDARY_SPECIES.casefold() in text:
            score = 0.52
        elif BENCHMARK_OUTGROUP_SPECIES.casefold() in text:
            score = 0.30
        elif BENCHMARK_SECOND_OUTGROUP_SPECIES.casefold() in text:
            score = 0.22
        elif "papilionidae" in text:
            score = 0.92
        elif "nymphalidae" in text:
            score = 0.36
        else:
            score = 0.05
        scores[str(label)] = score
    return scores


def _benchmark_text_embeddings(labels: list[str]) -> list[list[float]]:
    scores = _fake_label_scores(labels)
    return [[scores[label], sqrt(1.0 - scores[label] * scores[label])] for label in labels]


def _temporary_directories_left(output: Path) -> list[str]:
    if not output.exists():
        return []
    return sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_dir() and path.name.endswith(".tmp"))


def _taxon_row(
    accepted_taxon_key: str,
    scientific_name: str,
    family_key: str,
    family: str,
    genus_key: str,
    genus: str,
    retrieved_at: str,
) -> dict[str, object]:
    return {
        "registry_version": BENCHMARK_TAXONOMY_REGISTRY_VERSION,
        "scope_id": "benchmark-papilionoidea",
        "accepted_taxon_key": accepted_taxon_key,
        "scientific_name": scientific_name,
        "canonical_name": scientific_name,
        "rank": "SPECIES",
        "taxonomic_status": "accepted",
        "status": "accepted",
        "family_key": family_key,
        "family": family,
        "genus_key": genus_key,
        "genus": genus,
        "species_key": accepted_taxon_key,
        "species": scientific_name,
        "in_scope": True,
        "retrieved_at": retrieved_at,
    }


def _butterfly_bbox(detection_index: int) -> tuple[float, float, float, float]:
    tiled = (
        (4.0, 4.0, 28.0, 28.0),
        (36.0, 4.0, 60.0, 28.0),
        (4.0, 36.0, 28.0, 60.0),
        (36.0, 36.0, 60.0, 60.0),
    )
    if detection_index < len(tiled):
        return tiled[detection_index]
    offset = float((detection_index - len(tiled)) % 8)
    return (12.0 + offset, 12.0, 44.0 + offset, 44.0)


def _butterfly_record_count(records: int, *, butterfly_rate: float) -> int:
    return max(0, min(records, int(records * butterfly_rate)))


def _positive_int(value: object, *, name: str, allow_zero: bool = False) -> int:
    integer = int(value)
    if allow_zero and integer == 0:
        return integer
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _rate(value: object, *, name: str) -> float:
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _elapsed(start: float) -> float:
    return round(perf_counter() - start, 6)
