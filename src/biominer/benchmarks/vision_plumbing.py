from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter
import tracemalloc
from typing import Any, Mapping, Sequence

import polars as pl

from biominer.bioclip.candidate_sets import build_candidate_set
from biominer.bioclip.classification_modes import (
    DEFAULT_FAMILY_TOP_K,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    ClassificationMode,
    normalize_classification_mode,
)
from biominer.bioclip.object_runner import screen_object_detections
from biominer.bioclip.taxonomy_store import ButterflyTaxonomyStore
from biominer.detection.detector_base import DecodedImage, DetectionCandidate
from biominer.detection.pipeline import run_detection_pipeline
from biominer.detection.policy import DetectionPolicy, DetectionRunPolicy, detection_is_bioclip_eligible
from biominer.evidence.join import write_object_evidence_outputs
from biominer.registry.classification_table import (
    build_classification_artifact_frames,
    classification_artifact_paths,
)
from biominer.species.context import SpeciesContext
from biominer.storage.parquet import write_parquet


BENCHMARK_KIND = "vision_plumbing_model_free"
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

    def score(self, item: dict[str, Any], labels: tuple[str, ...]) -> dict[str, float]:
        self.score_calls += 1
        self.label_evaluations += len(labels)
        return _fake_label_scores(labels)

    def score_label_sets_batch(
        self,
        items: Sequence[dict[str, Any]],
        label_sets: Mapping[str, Sequence[str]],
    ) -> dict[str, list[dict[str, float]]]:
        self.batch_calls += 1
        self.label_evaluations += len(items) * sum(len(labels) for labels in label_sets.values())
        return {
            str(name): [_fake_label_scores(tuple(str(label) for label in labels)) for _item in items]
            for name, labels in label_sets.items()
        }


def run_vision_plumbing_benchmark(
    *,
    records: int,
    butterfly_rate: float,
    detections_per_butterfly: int,
    classification_mode: ClassificationMode = HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    taxonomy_candidate_table: str | Path | None = None,
    output_dir: str | Path,
    family_top_k: int = DEFAULT_FAMILY_TOP_K,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
) -> VisionPlumbingBenchmarkResult:
    record_count = _positive_int(records, name="records", allow_zero=True)
    butterfly_rate_value = _rate(butterfly_rate, name="butterfly_rate")
    detections_per_butterfly_count = _positive_int(detections_per_butterfly, name="detections_per_butterfly")
    mode = normalize_classification_mode(classification_mode)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    taxonomy_path = Path(taxonomy_candidate_table) if taxonomy_candidate_table is not None else output / "taxonomy_store"
    taxonomy_fixture_created = False

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
        if not taxonomy_path.exists():
            write_benchmark_taxonomy_store(taxonomy_path)
            taxonomy_fixture_created = True
        taxonomy_store = ButterflyTaxonomyStore.read(taxonomy_path)
        species_context = benchmark_species_context()
        candidate_set = build_candidate_set(species_context, records=canonical.to_dicts(), allow_single_target_fixture=True)
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
        scorer = BenchmarkBioClipScorer()
        scores_path = output / "object_bioclip_scores.parquet"
        score_result = screen_object_detections(
            canonical_records=canonical,
            detections=detection_result.frame,
            species_context=species_context,
            candidate_set=candidate_set,
            scorer=scorer,
            output_path=scores_path,
            classification_mode=mode,
            family_top_k=family_top_k,
            species_first_pass_top_k=species_first_pass_top_k,
            species_rerank_top_k=species_rerank_top_k,
            taxonomy_store=taxonomy_store if mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION else None,
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


def write_benchmark_taxonomy_store(root: str | Path) -> dict[str, Path]:
    output = Path(root)
    base = output.parent if output.suffix == ".parquet" else output
    base.mkdir(parents=True, exist_ok=True)
    taxa = benchmark_taxa_frame()
    classification_taxa, family_labels, species_labels, qa_findings, manifest = build_classification_artifact_frames(
        taxa,
        registry_manifest={"registry_version": BENCHMARK_TAXONOMY_REGISTRY_VERSION, "qa_status": "benchmark_fixture"},
    )
    paths = classification_artifact_paths(output)
    write_parquet(classification_taxa, paths["classification_taxa"])
    write_parquet(family_labels, paths["family_labels"])
    write_parquet(species_labels, paths["species_labels"])
    write_parquet(qa_findings, paths["qa_findings"])
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def benchmark_taxa_frame() -> pl.DataFrame:
    now = datetime.now(UTC).isoformat()
    rows = [
        _taxon_row("gbif:9401", BENCHMARK_PRIMARY_SPECIES, "gbif:9417", "Papilionidae", "gbif:9400", "Benchmarkus", now),
        _taxon_row("gbif:9402", BENCHMARK_SECONDARY_SPECIES, "gbif:9417", "Papilionidae", "gbif:9400", "Benchmarkus", now),
        _taxon_row("gbif:7001", BENCHMARK_OUTGROUP_SPECIES, "gbif:7017", "Nymphalidae", "gbif:7000", "Metricsus", now),
        _taxon_row("gbif:7002", BENCHMARK_SECOND_OUTGROUP_SPECIES, "gbif:7017", "Nymphalidae", "gbif:7000", "Metricsus", now),
    ]
    return pl.DataFrame(rows)


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
        "taxonomy_candidate_table": str(taxonomy_path),
        "taxonomy_fixture_created": taxonomy_fixture_created,
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


def _fake_label_scores(labels: Sequence[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label in labels:
        text = label.casefold()
        if "papilionidae" in text:
            score = 0.92
        elif "nymphalidae" in text:
            score = 0.36
        elif BENCHMARK_PRIMARY_SPECIES.casefold() in text:
            score = 0.86
        elif BENCHMARK_SECONDARY_SPECIES.casefold() in text:
            score = 0.52
        elif BENCHMARK_OUTGROUP_SPECIES.casefold() in text:
            score = 0.30
        elif BENCHMARK_SECOND_OUTGROUP_SPECIES.casefold() in text:
            score = 0.22
        else:
            score = 0.05
        scores[str(label)] = score
    return scores


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
