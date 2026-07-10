# Vision Workflow

The production vision path is object-first:

```text
canonical Flickr source records
-> YOLOE/YOLO26 object proposals
-> materialized score inputs
-> BioCLIP 2.5 scoring
-> joined object evidence and photo summaries
```

YOLOE/YOLO26 is only an object finder. The BioCLIP gate is configurable:

- `butterfly_like_only` is the legacy serial behavior: only `detection_status=detected`, `detector_label=butterfly_like` rows are scored.
- `exclude_hard_negative` is the rolling recall behavior: all detected non-hard-negative rows are scored as detector crops, and `no_detection` rows are scored as whole-image fallback when the image loaded.

Hard-negative and failed-image rows remain evidence rows but are not BioCLIP-scored. Production also avoids creating non-debug crop artifacts for hard-negative detections.

The current default classification mode is `target_scope_object_screening`. BioCLIP scores detector crops against target/scope candidate labels for screening evidence. Target-scope scoring can use registry-derived species candidates, but it is still screening evidence rather than taxonomic validation.

For target-scope local debugging, pass `--species-candidates data/registry/current/species_candidates.parquet` to `biominer vision score` or `biominer vision screen`.

Registry builds add GBIF-derived candidate tables for the family-first hierarchical classifier:

```text
butterfly_classification_taxa.parquet
butterfly_family_labels.parquet
butterfly_species_labels.parquet
```

The `hierarchical_butterfly_classification` mode is implemented when `--taxonomy-candidate-table` points at those artifacts. It is open classification, not target validation. BioCLIP first scores configured butterfly-family prompts, records family top 3, selects the top family, scores species prompts only within that selected GBIF family, records species top 20, then reranks all 20 first-pass species into species top 5. Prompt-template scores are mean-aggregated by taxon. The mode does not inject the run target species and does not treat geography as hard validation.

Hierarchical mode still obeys the configured BioCLIP gate. In rolling recall mode, moth-like, caterpillar, pupa, and generic insect-like detections are intentionally screened by BioCLIP rather than dropped at the detector boundary; hard-negative rows still remain evidence only.

The production default visual mode is `detector_crop`. Whole-image BioCLIP is available through explicit ablation/debug commands and through the rolling no-detection fallback; it is not an all-image default.

## Public Commands

The public stage tools are:

```bash
uv run biominer vision screen --help
uv run biominer vision rolling-screen --help
uv run biominer vision detect --help
uv run biominer vision score --help
uv run biominer vision ablate --help
```

Detection writes rows using the stable detection schema with source/photo join keys and object-level IDs. Scoring reads canonical records plus detections and writes BioCLIP object score rows. Ablation compares visual modes over the same input.

`vision detect` defaults to `--backend yoloe26`. The optional `--backend yolo26` path is inference-only compatibility for a user-provided coarse-object checkpoint. These stage commands are for local debugging; production runs are coordinated by `biominer run`.

```bash
uv run biominer vision detect \
  --input runs/local_debug/papilio_demoleus/canonical_source_records.parquet \
  --output runs/local_debug/papilio_demoleus/object_detections_yolo26.parquet \
  --backend yolo26 \
  --runtime-python "../YOLO26/venv/bin/python" \
  --checkpoint "../YOLO26/models/coarse-objects.pt"
```

Direct hierarchical scoring for local debugging uses the same taxonomy artifacts:

```bash
uv run biominer vision score \
  --input runs/local_debug/papilionoidea/canonical_source_records.parquet \
  --detections runs/local_debug/papilionoidea/object_detections_yoloe26.parquet \
  --species-context runs/local_debug/papilionoidea/species_context.json \
  --taxonomy-candidate-table data/registry/current \
  --classification-mode hierarchical_butterfly_classification \
  --family-top-k 3 \
  --species-first-pass-top-k 20 \
  --species-rerank-top-k 5 \
  --output runs/local_debug/papilionoidea/object_bioclip_scores.parquet
```

`--taxonomy-text-embedding-cache` is optional for hierarchical taxonomy labels. Target-scope caches such as `--candidate-text-embedding-cache` and `--object-image-embedding-cache` are not hierarchical taxonomy caches.

YOLO26 checkpoints must emit BioMiner coarse object labels or known legacy object aliases. Species-class checkpoints are rejected rather than remapped.

For a local detector-first run that keeps each stage as durable zstd part files, use `vision screen`. It runs one persistent YOLOE sidecar and one persistent BioCLIP sidecar, writes canonical/detection/score/joined/summary part directories, and deletes cached images only after the relevant outputs commit.

For the pipelined recall path, use `vision rolling-screen`. It plans deterministic 500-row image batches, uses YOLOE `image_path` transport by default, materializes `bioclip_score_inputs`, defaults to `--bioclip-gate-mode exclude_hard_negative`, enables `--score-no-detection-whole-image`, and keeps at most the configured image batches resident before commit-ordered cleanup:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer vision rolling-screen \
  --input runs/local_debug/papilionoidea/canonical_source_records.parquet \
  --output-dir runs/local_debug/papilionoidea/vision_rolling_screen \
  --species-context runs/local_debug/papilionoidea/species_context.json \
  --species-candidates data/registry/current/species_candidates.parquet \
  --vision-profile mac_m5pro_64gb \
  --device mps \
  --vision-batch-rows 500 \
  --bioclip-preprocess-workers 2 \
  --delete-images-after-commit
```

Supported visual modes are:

```text
whole_image
detector_crop
detector_crop_segmentation
```

`detector_crop_segmentation` is explicit but only produces segmentation-crop rows when masks are available. The first YOLOE-26 adapter emits boxes only, so regular detector crops are the default useful mode.

There is no image-enhancement mode in production, and BioMiner does not store reviewed boxes or a training dataset as part of this workflow.

## Debug And Runtime Commands

Runtime checks, model prefetch, smoke tests, previews, evaluations, and benchmarks live under `biominer dev vision`:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer dev vision yoloe26-runtime-check \
  --runtime-python "../YOLO26/venv/bin/python" \
  --checkpoint yoloe-26s-seg.pt \
  --device mps

PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer dev vision bioclip-runtime-check \
  --runtime-python "../BioCLIP25/venv/bin/python" \
  --device mps
```

These commands validate optional runtimes and maintained pipeline wiring. They are not the production entry point; production work is coordinated by `biominer run`.

## Benchmarks And Optimisation Checks

The deterministic plumbing benchmark exercises the detector-first pipeline with fake images, a fake detector, a fake BioCLIP scorer, and fake taxonomy artifacts. It is the normal regression check for counts, gating, batching, and report output because it does not require models or network access:

```bash
uv run biominer dev vision benchmark-plumbing \
  --records 1000 \
  --butterfly-rate 0.25 \
  --detections-per-butterfly 1 \
  --classification-mode hierarchical_butterfly_classification \
  --taxonomy-candidate-table tests/fixtures/taxonomy_store \
  --output-dir reports/vision_benchmarks/plumbing
```

The rolling benchmark matrix is also model-free. It compares YOLOE sidecar transport (`json_b64` versus `image_path`), accelerator concurrency, BioCLIP preprocessing workers, gate mode, and 250/500/1000-row batch sizes:

```bash
uv run biominer dev vision benchmark-rolling-matrix \
  --records 1000 \
  --output-dir reports/vision_benchmarks/rolling_matrix
```

The optional live benchmark is for Mac M5 Pro sidecar validation only. It fails clearly when the YOLOE or BioCLIP runtime path, taxonomy table, cache, or model is missing:

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

Adaptive batching is off by default. Enable `--adaptive-batching` only under memory pressure; conservative memory/device errors can reduce YOLO and BioCLIP batch sizes while non-memory errors still fail normally. Large hierarchical runs should prefer a validated `--taxonomy-text-embedding-cache` so species prompt embeddings are reused instead of recomputed.

## Evaluation And Review QA

Evaluation runs after detector/BioCLIP artifacts already exist. It is model-free: it reads object scores or joined object evidence plus reviewed labels and writes metrics, confusion matrices, calibration bins, review-error examples, and a Markdown summary.

```bash
uv run biominer evaluation classify \
  --object-evidence runs/local_debug/papilionoidea_hierarchical/object_evidence_joined.parquet \
  --reviewed-labels data/reviewed/papilionoidea_reviewed_labels.parquet \
  --output-dir reports/evaluation/papilionoidea_hierarchical
```

Use `--write-charts` only for local output directories when PNG charts are needed. The chart set is family confusion matrix, species accuracy by reviewed family, calibration reliability, and review reason counts.

Build a local hierarchical review queue from object evidence and optional photo summaries:

```bash
uv run biominer evaluation review-queue \
  --object-evidence runs/local_debug/papilionoidea_hierarchical/object_evidence_joined.parquet \
  --photo-summary runs/local_debug/papilionoidea_hierarchical/photo_evidence_summary.parquet \
  --output reports/review_queue.parquet
```

The production `summarize` stage writes the same kind of review-priority artifact automatically. Review queues rank uncertainty, conflict, and missing-score cases; they do not certify species truth.

Run Xie-style metrics as a reporting profile:

```bash
uv run biominer evaluation classify \
  --object-evidence runs/local_debug/papilionoidea_hierarchical/object_evidence_joined.parquet \
  --reviewed-labels data/reviewed/papilionoidea_reviewed_labels.parquet \
  --output-dir reports/evaluation/papilionoidea_hierarchical \
  --evaluation-profile xie_style_metrics_only
```

Xie-style here means macro/micro/per-family/top-k metrics for BioMiner outputs. It does not replace YOLOE-26, BioCLIP 2.5, GBIF candidate tables, or hierarchical candidate selection. BioCLIP scores remain candidate-set-relative, so reviewed labels are required for accuracy claims.

## Optional Runtimes

The main BioMiner package stays on Python 3.14. Heavy vision libraries run from Python 3.12 sidecar environments outside the repository:

```text
./BioMiner
./YOLO26/venv/bin/python
./YOLO26/models
./YOLO26/cache
./BioCLIP25/venv/bin/python
./BioCLIP25/models
./BioCLIP25/cache
```

Set `BIOMINER_BASE_PATH=/path/to/base` on macOS, WSL, or Ubuntu when the sibling folders are not next to the repository.

The Mac M5 Pro / 64 GB profile is `mac_m5pro_64gb`. It uses Apple MPS, YOLOE checkpoint `yoloe-26s-seg.pt`, YOLO image size `768`, detector batch size `16`, crop batch size `24`, crop target `336`, crop padding `0.08`, zstd Parquet part outputs, `parquet_part_rows=500`, and delete-after-commit image cleanup. Use `PYTORCH_ENABLE_MPS_FALLBACK=1` for runtime checks and sidecar runs.

Unit tests use fake detectors and fake scorers and must not require Ultralytics, CUDA, MPS, model downloads, or network access.

For the full hardware-specific command set and troubleshooting checklist, see `docs/m5pro_64gb_runbook.md`.

## Limitations

YOLOE-26 is zero-shot/open-vocabulary. It can miss butterflies and can confuse flowers, leaves, labels, or patterned material with insect-like objects. Low detector confidence can improve recall, but BioCLIP scores and evidence buckets must do the downstream filtering.

YOLOE/YOLO26 output must not be interpreted as species classification. It proposes boxes for BioCLIP; it does not validate family, genus, species, or occurrence identity.
