# M5 Pro Vision Performance Design Note

Scope: Phase 4 optimisation work for the detector-first YOLOE-26 plus BioCLIP 2.5 hierarchical butterfly pipeline. This note records the current hot-path audit and the optimisation boundary for a MacBook Pro M5 Pro / 64 GB workflow while preserving the S3/Postgres production shape.

## Target Profile

The `mac_m5pro_64gb` profile is the baseline local runtime:

```text
device: mps
YOLOE checkpoint: yoloe-26s-seg.pt
YOLO image size: 768
YOLO batch: 16
YOLO conf/iou/max-det: 0.20 / 0.50 / 8
BioCLIP model: hf-hub:imageomics/bioclip-2.5-vith14
BioCLIP crop batch: 24
crop target: 336
crop padding: 0.08
Parquet compression: zstd
cleanup: delete cached images/crops only after committed outputs
```

The profile lives in `src/biominer/detection/policy.py`. Heavy model dependencies stay in Python 3.12 sidecar runtimes; the main BioMiner Python 3.14 environment must remain light.

## Current Hot Paths

Detection loading:
`run_detection_pipeline(...)` loads images with bounded `ThreadPoolExecutor.map(..., buffersize=max_inflight_images)`, resizes images when needed, and submits detector batches according to `DetectionRunPolicy.detector_batch_size`.

YOLOE sidecar:
`YoloE26SidecarObjectDetector.detect_batch(...)` keeps a persistent subprocess and sends decoded RGB images as base64 JSON payloads. That is simple and testable, but it is likely the largest avoidable IPC cost on local high-throughput runs. The in-process optional YOLOE backend converts decoded images to PIL inside the sidecar/runtime path.

Crop generation:
Detection rows get crop metadata only for BioCLIP-eligible detections unless debug crop retention is enabled. Non-eligible detector rows remain evidence rows but normally do not materialise crop bytes. Object image embedding cache and direct crop materialisation write PPM files into temporary batch directories and remove them in `finally` blocks unless debug retention is requested.

BioCLIP worker:
BioCLIP sidecar requests already use image paths. `PersistentBioClipScorer` batches crop paths and supports label-set scoring, text embeddings, and image embeddings. This is the correct transport shape for large crop batches.

Hierarchical scoring:
The Phase 3 classifier keeps the required semantics: score configured families, select top family, score species top 20 inside that family, rerank all top 20 into top 5, and never inject the target species in hierarchical mode. Prompt-template scores are mean-aggregated by taxon.

Taxonomy lookup:
The taxonomy store loads classification taxa, family labels, and species labels once per score stage. Current lookup methods repeatedly filter and sort Polars frames for family candidates, species labels by family, and labels by accepted taxon key. The tables are small enough to load into memory, but repeated per-crop filtering is avoidable.

Text embedding reuse:
Taxonomy text embedding cache support exists as an optional input. It validates model/checkpoint, classification table version, prompt variant version, label hash, embedding dimension, and dtype. Direct prompt scoring remains the correctness fallback when no cache is supplied.

Parquet part writing:
Local `write_parquet(...)` uses a temporary file and atomic replace. Cloud detection and score stages write immutable Parquet part URIs before workstore shard registration and work item completion. This order is the production shape Phase 4 must preserve.

## Likely Bottlenecks

1. YOLOE sidecar base64 transport:
   Sending RGB bytes through JSON adds serialization, memory copying, and larger pipe payloads. Path-based batch input is the safest optimisation target if the sidecar can clean temporary files reliably.

2. BioCLIP prompt scoring:
   BioCLIP 2.5 Huge plus roughly 18,000 species labels can make repeated text encoding expensive. Cached taxonomy text embeddings should become a first-class optional production input for large family/species runs.

3. Taxonomy frame filtering:
   Repeated `filter/sort/select` calls against label tables are correct but wasteful. In-memory indices by family key and accepted taxon key are safe because classification tables are small compared with image/model memory.

4. MPS memory pressure:
   YOLO batch 16 and BioCLIP crop batch 24 are the performance target, not a guarantee under every image mix. Large detections, concurrent sidecars, and text embedding preparation can trigger MPS allocation failures.

5. Temporary crop lifecycle:
   Crop files must be created once per detection per batch, deduplicated by crop hash, and cleaned only after score rows are safely held or committed. Cleanup-before-commit is a correctness bug because failed writes must leave retryable inputs.

6. Observability gaps:
   Existing stage metrics cover core counts, but Phase 4 needs explicit skip counts, batch fallback metrics, selected-family/species summaries, cache-use flags, and throughput estimates.

## Memory Risks

- YOLOE and BioCLIP sidecars may both hold model weights on MPS.
- Base64 sidecar payloads can temporarily duplicate decoded image bytes in Python, JSON strings, and subprocess buffers.
- Taxonomy text embedding caches are larger than metadata-only tables, especially with float32 embeddings and multiple prompt templates.
- Whole-image or segmentation ablation modes can materially increase crop/image tensor memory; they must remain explicit debug modes.
- Adaptive batching must only catch conservative memory/device errors. Retrying arbitrary errors would hide data or code defects.

## Local vs Cloud Responsibilities

Local M5 Pro workflow:
- Tune sidecar IPC, crop materialisation, MPS batch sizes, and benchmark/report commands.
- Prefer model-free plumbing benchmarks for repeatable tests.
- Keep live MPS benchmarks optional and fail clearly when sidecar runtimes or models are missing.

Cloud/S3/Postgres workflow:
- Keep immutable Parquet part files.
- Register shards only after successful part writes.
- Complete work items only after shard registration.
- Never append to shared cloud files.
- Preserve deterministic work keys for every output-affecting setting.

Both paths:
- Load taxonomy tables once per score stage.
- Keep images and crops temporary.
- Preserve object/photo evidence rows for non-scored detections.
- Preserve `target_scope_object_screening` as the default.

## Implemented Phase 4 Optimisations

- Model-free plumbing benchmark: `biominer dev vision benchmark-plumbing` writes JSON/Markdown reports and uses only fake detectors/scorers.
- Optional live M5 Pro benchmark: `biominer dev vision benchmark-live-m5pro` validates sidecar runtimes, models, taxonomy tables, runtime settings, and MPS fallback state.
- Runtime profile validation: `mac_m5pro_64gb` keeps the target MPS profile and rejects invalid batch, crop, image-size, and adaptive override settings.
- YOLOE sidecar transport: `json_b64` remains compatible and `image_path` is available for lower local IPC overhead.
- Crop lifecycle: detector crops are materialised once per eligible detection batch where needed and cleaned only after score rows are safely held or written.
- Taxonomy text embedding cache: optional hierarchical taxonomy caches are validated against taxonomy and BioCLIP model metadata.
- Adaptive batching: opt-in detector and BioCLIP batch fallback retries only conservative memory/device failures and records retry metrics.
- Taxonomy lookup cache: classification tables are projected and indexed for repeated family/species lookups.
- Vision observability: `vision_stage_metrics.json` and `vision_stage_summary.md` expose skip counts, selected families/species, throughput, cache use, and batching behavior.
- Resumability keys: cloud detection and score work keys include output-affecting detector, classifier, taxonomy, prompt, top-k, and crop settings.

For operator commands, fallback settings, and troubleshooting, see `docs/m5pro_64gb_runbook.md`.

## Intentionally Not Implemented In Phase 4

- YOLO training or reviewed-box storage.
- Metadata anti-keywords as hard pre-visual discards.
- Permanent Flickr image archives.
- Hidden whole-image BioCLIP production scoring.
- Hidden segmentation production defaults.
- Broadening Flickr discovery/query behaviour.
- Adding PyTorch, OpenCLIP, Ultralytics, or Pillow-heavy runtime dependencies to the main required package.

## Do Not Regress

- YOLOE is an object detector only.
- BioCLIP is the family/species classifier.
- BioCLIP only receives YOLOE `butterfly_like` detections in production scoring.
- Family top 3 is scored across configured butterfly families.
- Species top 20 is constrained to the selected top family.
- Species top 5 is reranked from all top 20.
- Hierarchical mode never injects the target species.
- `target_scope_object_screening` remains backward-compatible and remains the default.
- Whole-image BioCLIP remains explicit ablation/debug behavior.
- Temporary images/crops are deleted only after committed outputs.
- Model-free tests stay deterministic, network-free, and free of real model dependencies.
