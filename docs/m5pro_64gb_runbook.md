# Mac M5 Pro 64 GB Vision Runbook

This runbook is the operator command set for BioMiner's `mac_m5pro_64gb` vision profile. It targets local Apple MPS runs while preserving the production S3/Postgres shape used by `biominer run`.

Related docs:

```text
docs/vision_workflow.md
docs/production_workflow.md
docs/gbif_classification_tables.md
docs/vision_performance_m5pro.md
```

## Profile Contract

The `mac_m5pro_64gb` profile is defined in `src/biominer/detection/policy.py` and should remain:

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

The main BioMiner environment stays on Python 3.14 and does not require PyTorch, Ultralytics, OpenCLIP, or model downloads. Heavy vision dependencies live in sibling Python 3.12 sidecar environments.

## Setup Assumptions

Expected local layout:

```text
BioMiner/
YOLO26/venv/bin/python
YOLO26/models/
YOLO26/cache/
BioCLIP25/venv/bin/python
BioCLIP25/cache/huggingface/
```

Set `BIOMINER_BASE_PATH` if those sibling folders are not next to the BioMiner checkout.

```bash
export BIOMINER_BASE_PATH=/path/to/workspace
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Runtime Checks

Validate the YOLOE sidecar:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer dev vision yoloe26-runtime-check \
  --runtime-python ../YOLO26/venv/bin/python \
  --checkpoint yoloe-26s-seg.pt \
  --device mps
```

Validate the BioCLIP sidecar:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer dev vision bioclip-runtime-check \
  --runtime-python ../BioCLIP25/venv/bin/python \
  --hf-cache-dir ../BioCLIP25/cache/huggingface \
  --device mps
```

Prefetch BioCLIP 2.5 Huge into the sidecar cache:

```bash
uv run biominer dev vision bioclip-prefetch-model \
  --runtime-python ../BioCLIP25/venv/bin/python \
  --hf-cache-dir ../BioCLIP25/cache/huggingface \
  --model-name imageomics/bioclip-2.5-vith14
```

Prefetch or validate the YOLOE checkpoint through the sidecar:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer dev vision yoloe26-prefetch \
  --runtime-python ../YOLO26/venv/bin/python \
  --checkpoint yoloe-26s-seg.pt \
  --device mps
```

## Model-Free Plumbing Benchmark

Use this first after code changes. It runs fake detector/scorer components, does not need model runtimes, and writes JSON and Markdown reports under the output directory.

```bash
uv run biominer dev vision benchmark-plumbing \
  --records 1000 \
  --butterfly-rate 0.25 \
  --detections-per-butterfly 1 \
  --classification-mode hierarchical_butterfly_classification \
  --taxonomy-candidate-table tests/fixtures/taxonomy_store \
  --output-dir reports/vision_benchmarks/plumbing
```

Expected outputs:

```text
reports/vision_benchmarks/plumbing/benchmark_metrics.json
reports/vision_benchmarks/plumbing/benchmark_summary.md
```

## Optional Live M5 Pro Benchmark

Use this only after the runtime checks pass. It is intentionally not part of the normal test suite.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer dev vision benchmark-live-m5pro \
  --input runs/local_debug/papilio_demoleus/canonical_source_records.parquet \
  --taxonomy-candidate-table data/registry/current \
  --vision-runtime-python ../YOLO26/venv/bin/python \
  --bioclip-runtime-python ../BioCLIP25/venv/bin/python \
  --hf-cache-dir ../BioCLIP25/cache/huggingface \
  --checkpoint yoloe-26s-seg.pt \
  --device mps \
  --limit 100 \
  --output-dir reports/vision_benchmarks/m5pro_live
```

For lower IPC overhead after validating the sidecar path mode:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer dev vision benchmark-live-m5pro \
  --input runs/local_debug/papilio_demoleus/canonical_source_records.parquet \
  --taxonomy-candidate-table data/registry/current \
  --vision-runtime-python ../YOLO26/venv/bin/python \
  --bioclip-runtime-python ../BioCLIP25/venv/bin/python \
  --hf-cache-dir ../BioCLIP25/cache/huggingface \
  --checkpoint yoloe-26s-seg.pt \
  --yolo-sidecar-transport image_path \
  --device mps \
  --limit 100 \
  --output-dir reports/vision_benchmarks/m5pro_live_path_transport
```

## Local Hierarchical Run

This is the recommended local detector-first hierarchical command. It uses local storage and SQLite, selects the rolling recall worker, preserves BioCLIP score-input gating, and deletes cached images only after committed outputs.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer run \
  --taxon "Papilionoidea" \
  --rank family \
  --registry-dir data/registry/current \
  --output-prefix runs/local_debug/papilionoidea_hierarchical \
  --storage-backend local \
  --workstore-backend sqlite \
  --vision-backend yoloe26 \
  --vision-profile mac_m5pro_64gb \
  --vision-worker rolling \
  --classification-mode hierarchical_butterfly_classification \
  --taxonomy-candidate-table data/registry/current \
  --device mps \
  --family-top-k 3 \
  --species-first-pass-top-k 20 \
  --species-rerank-top-k 5 \
  --delete-images-after-commit
```

If a taxonomy text embedding cache has already been prepared and validated for the same classification table, prompt variant, BioCLIP model, and checkpoint, add:

```bash
  --taxonomy-text-embedding-cache data/registry/current/butterfly_taxonomy_text_embeddings.parquet
```

## Post-Run Evaluation

After a local hierarchical run writes `object_evidence_joined.parquet`, evaluate it against reviewed labels:

```bash
uv run biominer evaluation classify \
  --object-evidence runs/local_debug/papilionoidea_hierarchical/object_evidence_joined.parquet \
  --reviewed-labels data/reviewed/papilionoidea_reviewed_labels.parquet \
  --output-dir reports/evaluation/papilionoidea_hierarchical \
  --write-charts
```

The evaluation step is model-free and should run in the main Python 3.14 environment. It reports family top1/top3, selected-family accuracy, species top1/top5/top20, MRR, family/species confusion matrices, heuristic calibration bins, review-error examples, and optional PNG charts. Human-reviewed labels are required for biological accuracy claims; synthetic fixtures only prove arithmetic, schema, and regression behavior.

Build or refresh the local review queue when inspecting artifacts outside a full `biominer run summarize` stage:

```bash
uv run biominer evaluation review-queue \
  --object-evidence runs/local_debug/papilionoidea_hierarchical/object_evidence_joined.parquet \
  --photo-summary runs/local_debug/papilionoidea_hierarchical/photo_evidence_summary.parquet \
  --output reports/review_queue.parquet
```

The queue ranks low-margin, conflicting, missing-score, hard-negative, metadata-conflict, multi-object, and geospatial-prior cases for human inspection. It is not truth data.

Run Xie-style metrics only as an evaluation profile:

```bash
uv run biominer evaluation classify \
  --object-evidence runs/local_debug/papilionoidea_hierarchical/object_evidence_joined.parquet \
  --reviewed-labels data/reviewed/papilionoidea_reviewed_labels.parquet \
  --output-dir reports/evaluation/papilionoidea_hierarchical \
  --evaluation-profile xie_style_metrics_only
```

Xie-style here means report macro/micro/per-family/top-k metrics over BioMiner outputs. It does not replace the detector-first architecture, and BioCLIP scores remain candidate-set-relative rather than calibrated probabilities.

## Conservative Fallback Run

Use this when MPS memory pressure appears during live runs. Adaptive batching is opt-in and only retries conservative memory/device failures.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer run \
  --taxon "Papilionoidea" \
  --rank family \
  --registry-dir data/registry/current \
  --output-prefix runs/local_debug/papilionoidea_hierarchical_safe \
  --storage-backend local \
  --workstore-backend sqlite \
  --vision-backend yoloe26 \
  --vision-profile mac_m5pro_64gb \
  --vision-worker rolling \
  --classification-mode hierarchical_butterfly_classification \
  --taxonomy-candidate-table data/registry/current \
  --device mps \
  --yolo-batch 8 \
  --bioclip-batch 12 \
  --adaptive-batching \
  --delete-images-after-commit
```

## Cloud S3/Postgres Production Run

The same classifier semantics apply in cloud production: immutable Parquet parts, Postgres workstore claims, rolling 500-image work keys, `exclude_hard_negative` BioCLIP score-input gating, and no-detection whole-image fallback when the image loaded.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer run \
  --taxon "Papilionoidea" \
  --rank family \
  --registry-dir s3://biominer/biominer/registry/current \
  --output-prefix s3://biominer/biominer/runs/papilionoidea_hierarchical \
  --storage-backend s3 \
  --workstore-backend postgres \
  --vision-backend yoloe26 \
  --vision-profile mac_m5pro_64gb \
  --vision-worker rolling \
  --classification-mode hierarchical_butterfly_classification \
  --taxonomy-candidate-table s3://biominer/biominer/registry/current \
  --device mps \
  --family-top-k 3 \
  --species-first-pass-top-k 20 \
  --species-rerank-top-k 5 \
  --delete-images-after-commit
```

## Expected Artifacts

Local and cloud runs write stage outputs and reports under the configured prefix. Visual stages use zstd Parquet part files:

```text
evidence/stage=detect_objects/run_id=<run_id>/worker=<worker_id>/part=<part_id>.parquet
evidence/stage=score_bioclip/run_id=<run_id>/worker=<worker_id>/part=<part_id>.parquet
vision_stage_metrics.json
vision_stage_summary.md
manifest.json
```

Important counters to inspect after a run:

```text
detection_counts.images_seen
detection_counts.images_loaded
detection_counts.detections
detection_counts.crops_created
bioclip_counts.objects_scored
bioclip_counts.detector_crops_scored
bioclip_counts.whole_images_scored
metrics.bioclip_score_inputs
metrics.bioclip_score_inputs_per_image
metrics.butterfly_like_detections
metrics.eligible_bioclip_detections
metrics.bioclip_gate_mode
metrics.selected_family_counts
metrics.species_top1_counts
metrics.adaptive_batching_enabled
metrics.detector_batch_retries
metrics.bioclip_batch_retries
```

In rolling recall mode, `metrics.bioclip_score_inputs` is the denominator for BioCLIP work: non-hard-negative detections plus configured no-detection fallback rows. `metrics.eligible_bioclip_detections` remains the legacy detector-policy count and should not be used as the rolling score-input denominator.

## Failure Modes

Missing sidecar Python:
Runtime checks and live benchmarks fail before loading records. Fix the sidecar venv path or `BIOMINER_BASE_PATH`.

Missing BioCLIP cache:
Run `bioclip-prefetch-model` or point `--hf-cache-dir` at the populated sidecar cache. Do not commit model caches.

YOLOE checkpoint rejected:
Use `yoloe-26s-seg.pt` or another supported YOLOE-26 segmentation checkpoint. Species-class checkpoints are not valid object detectors for this path.

MPS memory errors:
Use the conservative fallback command, lower `--yolo-batch` or `--bioclip-batch`, and enable `--adaptive-batching`. Keep `PYTORCH_ENABLE_MPS_FALLBACK=1` set for Apple Silicon runs.

Unexpected all-image BioCLIP scoring:
Check requested ablation modes and gate settings. Production defaults to `detector_crop`; whole-image and segmentation scoring are explicit debug/ablation modes except for the rolling no-detection fallback.

Too many non-butterfly crops:
Rolling recall intentionally materialises crops for non-hard-negative detections, including moth-like, caterpillar, pupa, and generic insect-like labels. It should not materialise crops for hard-negative detections. Inspect `bioclip_score_inputs`, `hard_negative_detections`, `bioclip_gate_mode`, and score row counts.

Cleanup did not happen:
Cached images and crops are deleted only after committed outputs. If detection, score, Parquet write, or workstore registration fails, cached inputs can remain retryable by design.

## Invariants

- YOLOE is an object detector, not a taxonomic classifier.
- Rolling production BioCLIP receives detected non-hard-negative crops and no-detection whole-image fallback rows, but never hard-negative rows.
- Legacy `butterfly_like_only` scoring remains available for serial runs and ablations.
- Hierarchical mode scores family top 3, constrains species top 20 to the selected family, and reranks all 20 into top 5.
- Hierarchical mode does not inject the target species.
- `target_scope_object_screening` remains the default classification mode.
- Hard-negative detections remain evidence but are not BioCLIP species-scored.
- Images and crops are temporary and must not become a permanent Flickr image archive.
