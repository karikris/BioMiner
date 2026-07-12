from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
import logging
from pathlib import Path

import polars as pl

from biominer.bioclip.bioclip import PersistentBioClipScorer
from biominer.bioclip.candidate_sets import build_candidate_set
from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.bioclip.model_registry import BioClipRuntime, ModelConfig
from biominer.bioclip.object_runner import EphemeralCropBioClipScorer
from biominer.bioclip.path_taxonomy_store import PathTaxonomyStore
from biominer.bioclip.taxonomy_embedding_cache import TaxonomyTextEmbeddingIndex
from biominer.cli import BIOCLIP_25_HUGE_REPO_ID, BIOCLIP_25_HUGE_REVISION
from biominer.detection.policy import DetectionPolicy, vision_runtime_settings
from biominer.detection.segmentation import make_segmenter
from biominer.detection.yoloe26_detector import YoloE26SidecarObjectDetector, default_yoloe26_prompts
from biominer.species.context import SpeciesContext
from biominer.vision.gates import BioClipGatePolicy
from biominer.vision.rolling_worker import (
    BioCLIPWorker,
    CommitWorker,
    ImageStager,
    RollingVisionWorker,
    RollingVisionWorkerSettings,
    ScoreInputMaterializer,
    YOLOWorker,
    load_staged_or_cached_image,
)


def main() -> int:
    args = _parser().parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _configure_logging(output / "vision_workflow.log")
    logger = logging.getLogger("biominer.vision.local_runner")
    started = datetime.now(UTC)
    records = pl.read_parquet(args.input)
    context = SpeciesContext.read_json(args.species_context)
    candidates = build_candidate_set(context, allow_single_target_fixture=True)
    taxonomy_store = PathTaxonomyStore.read(args.registry_dir)
    settings = vision_runtime_settings(args.vision_profile).with_overrides(
        device=args.device,
        yolo_sidecar_transport="image_path",
        parquet_part_rows=args.batch_rows,
    )
    detector = YoloE26SidecarObjectDetector(
        runtime_python=args.yolo_runtime_python,
        checkpoint=settings.yolo_checkpoint,
        device=settings.device,
        imgsz=settings.yolo_imgsz,
        conf=settings.yolo_conf,
        iou=settings.yolo_iou,
        max_det=settings.yolo_max_det,
        prompt_classes=default_yoloe26_prompts(include_hard_negative_prompts=True),
        transport=settings.yolo_sidecar_transport,
    )
    scorer = PersistentBioClipScorer(
        runtime=_bioclip_runtime(Path(args.bioclip_runtime_python)),
        hf_cache_dir=args.hf_cache_dir,
        device=settings.device,
    )
    stager = ImageStager(output_dir=output, cache_root=args.cache_root)
    try:
        crop_scorer = EphemeralCropBioClipScorer(
            scorer=scorer,
            image_loader=load_staged_or_cached_image,
            temp_dir=args.crop_temp_dir,
            crop_padding_ratio=settings.crop_padding_ratio,
            crop_target_px=settings.crop_target_px,
            model_id=BIOCLIP_25_HUGE_REPO_ID,
            model_version="bioclip2_5_huge",
            model_checkpoint=BIOCLIP_25_HUGE_REVISION,
            retain_debug_crops=False,
            debug_crop_limit=settings.debug_crop_limit,
            segmenter=make_segmenter("none"),
        )
        gate = BioClipGatePolicy(mode="exclude_hard_negative", score_no_detection_whole_image=True)
        embedding_frame = pl.read_parquet(args.taxonomy_text_embedding_cache)
        embedding_index = TaxonomyTextEmbeddingIndex.from_frame(
            embedding_frame,
            taxonomy_store=taxonomy_store,
            model_id=crop_scorer.model_id,
            model_checkpoint=crop_scorer.model_checkpoint,
        )
        rolling_settings = RollingVisionWorkerSettings(
            vision_batch_rows=args.batch_rows,
            heartbeat_interval_seconds=args.heartbeat_seconds,
            bioclip_gate_mode=gate.normalized_mode.value,
            score_no_detection_whole_image=True,
        )
        worker = RollingVisionWorker(
            settings=rolling_settings,
            image_stage=stager,
            detection_stage=YOLOWorker(
                detector=detector,
                output_dir=output,
                image_loader=load_staged_or_cached_image,
                detection_policy=settings.to_detection_policy(DetectionPolicy(backend="yoloe26")),
                run_policy=settings.to_detection_run_policy(),
            ),
            score_input_stage=ScoreInputMaterializer(
                output_dir=output,
                image_loader=load_staged_or_cached_image,
                gate_policy=gate,
                crop_padding_ratio=settings.crop_padding_ratio,
                crop_target_px=settings.crop_target_px,
            ),
            score_stage=BioCLIPWorker(
                species_context=context,
                candidate_set=candidates,
                scorer=crop_scorer,
                output_dir=output,
                bioclip_batch_size=settings.crop_batch_size,
                adaptive_batching=settings.adaptive_batching,
                min_bioclip_batch_size=settings.min_crop_batch_size,
                classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
                path_taxonomy_store=taxonomy_store,
                taxonomy_text_embedding_index=embedding_index,
            ),
            commit_stage=CommitWorker(
                output_dir=output,
                species_context=context,
                delete_images_after_commit=True,
            ),
        )
        logger.info("local_vision_run records=%d candidates=%d", records.height, len(candidates.species_candidates))
        result = worker.run(records)
    finally:
        stager.close()
        scorer.close()
        detector.close()
    manifest = {
        "status": result.status,
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "input": str(args.input),
        "records": records.height,
        "started_at": started.isoformat(),
        "ended_at": datetime.now(UTC).isoformat(),
        "candidate_set_id": candidates.candidate_set_id,
        "species_candidate_count": len(candidates.species_candidates),
        "registry_dir": str(args.registry_dir),
        "taxonomy_classification_fingerprint": taxonomy_store.classification_fingerprint,
        "taxonomy_hierarchy_fingerprint": taxonomy_store.hierarchy_fingerprint,
        "taxonomy_embedding_cache": str(args.taxonomy_text_embedding_cache),
        "taxonomy_embedding_cache_fingerprint": embedding_index.cache_fingerprint,
        "rolling_settings": asdict(rolling_settings),
        "metrics": result.metrics,
        "part_outputs": list(result.part_outputs),
        "log": str(output / "vision_workflow.log"),
    }
    (output / "vision_workflow_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _bioclip_runtime(runtime_python: Path) -> BioClipRuntime:
    model = ModelConfig(
        model_id="bioclip2_5_huge",
        display_name="BioCLIP 2.5 Huge",
        role="preferred",
        status="use_if_available",
        task="biology image-text classification and embedding",
        model_name=BIOCLIP_25_HUGE_REPO_ID,
        checkpoint=BIOCLIP_25_HUGE_REVISION,
        package_name="open_clip_torch",
        package_version="3.3.0",
        model_hash=f"hf-revision:{BIOCLIP_25_HUGE_REVISION}",
    )
    return BioClipRuntime(
        model=model,
        home=runtime_python.parent.parent,
        venv_python=runtime_python,
        package_version="3.3.0",
        available=True,
    )


def _configure_logging(path: Path) -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local rolling YOLOE + BioCLIP batch")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--species-context", required=True)
    parser.add_argument("--registry-dir", required=True)
    parser.add_argument("--taxonomy-text-embedding-cache", required=True)
    parser.add_argument("--yolo-runtime-python", required=True)
    parser.add_argument("--bioclip-runtime-python", required=True)
    parser.add_argument("--hf-cache-dir", required=True)
    parser.add_argument("--cache-root", default="data/cache/images")
    parser.add_argument("--crop-temp-dir", default="data/cache/object_crops")
    parser.add_argument("--vision-profile", default="mac_m5pro_64gb")
    parser.add_argument("--device", default="mps", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--batch-rows", type=int, default=500)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
