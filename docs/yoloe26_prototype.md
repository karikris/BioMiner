# YOLOE-26 Detector-First Prototype

This prototype uses YOLOE-26 only as an open-vocabulary object proposal backend. BioCLIP 2.5 Huge provides target/scope visual screening until the guarded hierarchical classifier is implemented.

The current default classification mode is `target_scope_object_screening`: BioCLIP scores YOLOE `butterfly_like` detector crops against target/scope candidate labels. Columns such as `family_top3`, `species_top20`, and `species_top5` are screening fields, not a completed family-first hierarchical classifier. The reserved `hierarchical_butterfly_classification` mode is guarded until the GBIF classification-table workflow is implemented.

## Setup

Install the optional Python 3.12 runtimes outside the BioMiner repository:

```bash
cd ./BioMiner
bash scripts/setup_yoloe26_user_py312.sh
bash scripts/setup_bioclip25_user_py312.sh
```

Default sibling layout from the base directory:

```text
./BioMiner
./YOLO26/venv/bin/python
./YOLO26/models
./YOLO26/cache
./BioCLIP25/venv/bin/python
./BioCLIP25/models
./BioCLIP25/cache
```

Commands below are run from `./BioMiner`. Set `BIOMINER_BASE_PATH=/path/to/base` if the sibling folders live somewhere else.

## Runtime Checks

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer dev vision yoloe26-runtime-check \
  --runtime-python "../YOLO26/venv/bin/python" \
  --checkpoint yoloe-26s-seg.pt \
  --device mps

PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer dev vision bioclip-runtime-check \
  --runtime-python "../BioCLIP25/venv/bin/python" \
  --hf-cache-dir "../BioCLIP25/cache/huggingface" \
  --device mps
```

## Prefetch

```bash
uv run biominer dev vision yoloe26-prefetch \
  --runtime-python "../YOLO26/venv/bin/python" \
  --checkpoint yoloe-26s-seg.pt \
  --device auto

uv run biominer dev vision bioclip-prefetch-model \
  --runtime-python "../BioCLIP25/venv/bin/python" \
  --hf-cache-dir "../BioCLIP25/cache/huggingface" \
  --model-name imageomics/bioclip-2.5-vith14
```

## Run Detection Only

```bash
uv run biominer vision detect \
  --backend yoloe26 \
  --runtime-python "../YOLO26/venv/bin/python" \
  --checkpoint yoloe-26s-seg.pt \
  --input runs/local_debug/papilio_demoleus/canonical_source_records.parquet \
  --output runs/local_debug/papilio_demoleus/object_detections_yoloe26.parquet \
  --device auto \
  --conf 0.20 \
  --iou 0.50 \
  --max-det 8
```

Default YOLOE prompts are mapped into existing BioMiner coarse detector labels:

```text
butterfly, butterfly wing, pinned butterfly specimen, butterfly specimen, lepidoptera -> butterfly_like
moth -> moth_like
caterpillar -> caterpillar
chrysalis, pupa -> pupa
insect -> insect_like
flower, leaf, person, hand, drawing, painting, logo, text, sign, museum label -> hard_negative
```

## One-Command Prototype Run

Run an integrated local detector-crop screen with the Mac M5 Pro profile:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer vision screen \
  --input runs/local_debug/papilio_demoleus/canonical_source_records.parquet \
  --output-dir runs/local_debug/papilio_demoleus/vision_screen \
  --species-context runs/local_debug/papilio_demoleus/species_context.json \
  --species-candidates data/registry/current/species_candidates.parquet \
  --vision-profile mac_m5pro_64gb \
  --device mps \
  --delete-images-after-commit
```

This writes zstd part files under `canonical_source_records/`, `object_detections/`, `object_bioclip_scores/`, `object_evidence_joined/`, and `photo_evidence_summary/`. Cached images are deleted only after the relevant committed parts exist.

The older dev wrapper remains useful for bounded prototype debugging:

```bash
uv run biominer dev vision yoloe26-prototype-run \
  --input runs/local_debug/papilio_demoleus/canonical_source_records.parquet \
  --species-context runs/local_debug/papilio_demoleus/species_context.json \
  --species-candidates data/registry/current/species_candidates.parquet \
  --output-dir reports/yoloe26_prototype/example \
  --vision-profile mac_m5pro_64gb \
  --vision-runtime-python "../YOLO26/venv/bin/python" \
  --bioclip-runtime-python "../BioCLIP25/venv/bin/python" \
  --hf-cache-dir "../BioCLIP25/cache/huggingface" \
  --limit 10
```

Expected outputs:

```text
object_detections_yoloe26.parquet
object_bioclip_scores_yoloe26.parquet
object_evidence_joined_yoloe26.parquet
photo_evidence_summary_yoloe26.parquet
yoloe26_metrics.json
yoloe26_run_manifest.json
yoloe26_summary.md
```

## Smoke Test

Use a manual image when available:

```bash
uv run biominer dev vision yoloe26-smoke \
  --runtime-python "../YOLO26/venv/bin/python" \
  --checkpoint yoloe-26s-seg.pt \
  --device auto \
  --image /path/to/manual_test_image.jpg \
  --output-dir reports/yoloe26_smoke
```

Without `--image`, the command uses a synthetic placeholder and only validates runtime plumbing.

## Limits

- YOLOE-26 is zero-shot/open-vocabulary. It can miss butterflies and can confuse leaf, flower, textile, or label regions with insects.
- YOLOE-26 is not taxonomic validation. It only proposes object boxes.
- Only YOLOE `butterfly_like` detections are sent to BioCLIP in the production detector-first path.
- BioCLIP 2.5 Huge remains the target/scope screening scorer until the guarded hierarchical classifier is implemented.
- Whole-image BioCLIP is an explicit ablation/debug mode, not the production default.
- Metrics from `yoloe26-prototype-run` are labelled heuristic unless reviewed ground truth is supplied.
- Model files, caches, downloaded Flickr images, and generated Parquet outputs must not be committed.

The next data-engineering step is to review YOLOE-26 proposal quality through object evidence summaries, tune prompt/confidence policy, and keep any manual review decisions in evidence QA outputs rather than detector training artifacts.
