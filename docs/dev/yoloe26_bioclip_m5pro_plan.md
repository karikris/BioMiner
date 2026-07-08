# YOLOE-26 + BioCLIP M5 Pro Implementation Map

This note maps the current vision path before implementing the `mac_m5pro_64gb`
production profile. The target production workflow is detector-first: YOLOE-26
proposes butterfly objects, and only eligible butterfly detections are sent to
BioCLIP 2.5 for biological scoring.

## Current Defaults

- `DetectionPolicy` defaults to `backend="yoloe26"`, score threshold `0.20`,
  NMS IoU `0.50`, `max_boxes_per_image=8`, `crop_target_px=336`, and
  `crop_padding_ratio=0.12`.
- `DetectionRunPolicy` defaults to `download_workers=4`,
  `decode_workers=4`, `detector_workers=1`, `max_inflight_images=32`,
  `max_inflight_crops=96`, `detector_batch_size=4`, `crop_batch_size=24`,
  and `parquet_batch_rows=10000`.
- `MAC_M5PRO_64GB_PROFILE` currently overrides only image max side, crop
  target, debug crop retention, worker counts, inflight limits, and batch
  counts. It still uses detector batch size `4` and crop padding `0.12`.
- `config/vision_profiles/mac_m5pro_64gb.json` mirrors those older defaults
  and does not carry device, YOLOE checkpoint, YOLOE image size, BioCLIP model,
  BioCLIP top-k, Parquet compression, or delete-after-commit settings.
- `biominer run` defaults to S3 storage and Postgres workstore, but it does
  not expose `--vision-profile`, YOLOE image size or detector batch settings,
  BioCLIP crop batch/top-k settings, crop padding, Parquet compression, or
  delete-after-commit options.

## Existing Detector-First Gate

- `detection_is_bioclip_eligible()` in `src/biominer/detection/policy.py`
  requires `detection_status == "detected"` and a detector label in
  `DetectionPolicy.bioclip_eligible_labels`, currently only
  `("butterfly_like",)`.
- `screen_object_detections()` in `src/biominer/bioclip/object_runner.py`
  skips non-eligible detections before calling the object scorer.
- `materialize_detector_crop_inputs()` also filters through
  `detection_is_bioclip_eligible()` before materialising crop files.
- `enqueue_bioclip_work_from_detection_shards()` in
  `src/biominer/bioclip/cloud_work.py` only enqueues score work for eligible
  detection rows.
- Tests already cover the key gate in `tests/test_object_bioclip_pipeline.py`
  and `tests/test_cloud_bioclip_work.py`.

## One-Object Scoring Hot Path

- `ObjectBioClipScorer` exposes only `score(item, labels)`.
- `_score_detection()` calls the scorer separately for family labels, genus
  labels, species labels, and rerank species labels for each object.
- `screen_object_detections()` loops detection-by-detection and buffers one
  score row at a time.
- `run_cloud_bioclip_batch()` loops work-item-by-work-item and calls
  `_score_detection()` once per item.
- `PersistentBioClipScorer` already supports `score_batch()` and
  `score_label_sets_batch()`, and the worker caches text features per label
  tuple, but the object evidence path does not yet use that batch interface.

## YOLOE Sidecar Reload Point

- `YoloE26ObjectDetector` is direct in-process and loads YOLOE once in its
  constructor.
- `YoloE26SidecarObjectDetector.detect_batch()` currently calls
  `subprocess.run([runtime_python, "-m", "biominer.detection.yoloe26_detector"])`
  for every batch.
- `_run_sidecar()` reads a single JSON request from stdin, constructs a
  `YoloE26ObjectDetector`, runs one batch, prints one JSON response, and exits.
- This means sidecar mode launches a new process and reloads the YOLOE model
  for each detector batch.

## Production Visual Modes

- `ProductionRunRequest.bioclip_ablation_mode` defaults to `"detector_crop"`,
  but `bioclip_ablation_modes` defaults to `OBJECT_VISUAL_MODES`, which is
  `("whole_image", "detector_crop", "detector_crop_segmentation")`.
- `_request_bioclip_modes()` prefers `bioclip_ablation_modes`, so production
  currently requests all object visual modes unless the request overrides that
  tuple.
- `vision ablate` defaults to all modes and should stay able to request
  `whole_image`, `detector_crop`, and `detector_crop_segmentation` explicitly.

## Parquet Output Gap

- Local `write_parquet()` in `src/biominer/storage/parquet.py` writes through
  a temporary path and atomic replace, but it does not accept or record a
  compression option.
- Local detection and score pipelines write temporary batch files, read them
  back into memory with Polars, then write a final single Parquet file.
- `LocalStorageBackend.write_parquet_shard()` delegates to the same local
  writer and has no compression argument.
- `S3StorageBackend.write_parquet_batches()` uses PyArrow with zstd, but
  `S3StorageBackend.write_parquet_shard()` writes a single frame without an
  explicit compression setting.
- Cloud production writes one shard per claimed batch and registers it after a
  successful storage write, but there is no shared part-file writer API with
  immutable part paths, compression metadata, or part commit callbacks.

## Image Deletion Gap

- `cache_image_from_url()` stores downloaded images under
  `data/cache/images` by content hash and never deletes them.
- `load_decoded_image_from_record()` decodes the cached path and returns a
  `DecodedImage` with `source_uri` set to that path, but it does not expose a
  lease or cleanup callback.
- Temporary BioCLIP crop files are deleted by `EphemeralCropBioClipScorer`
  after each `score()` call unless debug retention is enabled.
- Detection debug crops are retained only when `retain_debug_crops` is true.
- No current workflow delays source image deletion until both detection rows
  and the relevant BioCLIP score rows have been durably written and registered.
  The integrated screen command needs a commit-aware cleanup queue or lease so
  failed part writes leave cached source images available for retry.

## Implementation Order

1. Add a typed runtime settings object that keeps compatibility with
   `DetectionPolicy`, `DetectionRunPolicy`, and `RuntimeProfile`.
2. Update code and JSON profile defaults for `mac_m5pro_64gb`.
3. Wire `biominer run` to accept and surface the vision runtime settings while
   preserving S3/Postgres as production defaults.
4. Change production default BioCLIP modes to detector crop only, keeping
   explicit ablation tools unchanged.
5. Add a persistent YOLOE sidecar protocol and make sidecar detection reuse
   one process across detector batches.
6. Refactor object crop materialisation and BioCLIP scoring to use crop
   batches and `PersistentBioClipScorer.score_label_sets_batch()`.
7. Add zstd immutable Parquet part writing and commit-aware cached image
   deletion for the integrated local screen workflow.
