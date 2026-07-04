# YOLOE-26 Detector-First Prototype

This prototype uses YOLOE-26 only as an open-vocabulary object proposal backend. BioCLIP 2.5 Huge remains BioMiner's biological classifier and species scorer.

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
uv run biominer vision yoloe26-runtime-check \
  --runtime-python "../YOLO26/venv/bin/python" \
  --checkpoint yoloe-26s-seg.pt \
  --device auto

uv run biominer vision bioclip-runtime-check \
  --runtime-python "../BioCLIP25/venv/bin/python" \
  --hf-cache-dir "../BioCLIP25/cache/huggingface" \
  --device auto
```

## Prefetch

```bash
uv run biominer vision yoloe26-prefetch \
  --runtime-python "../YOLO26/venv/bin/python" \
  --checkpoint yoloe-26s-seg.pt \
  --device auto

uv run biominer vision bioclip-prefetch-model \
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
  --input staging/species_runs/example/filtered.parquet \
  --output staging/species_runs/example/object_detections_yoloe26.parquet \
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

Run a bounded 10-image detector-crop prototype:

```bash
uv run biominer vision yoloe26-prototype-run \
  --input staging/species_runs/example/filtered.parquet \
  --species-context staging/species_runs/example/species_context.json \
  --output-dir reports/yoloe26_prototype/example \
  --vision-runtime-python "../YOLO26/venv/bin/python" \
  --bioclip-runtime-python "../BioCLIP25/venv/bin/python" \
  --hf-cache-dir "../BioCLIP25/cache/huggingface" \
  --device auto \
  --checkpoint yoloe-26s-seg.pt \
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
uv run biominer vision yoloe26-smoke \
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
- BioCLIP 2.5 Huge remains the family/genus/species scorer.
- Metrics from `yoloe26-prototype-run` are labelled heuristic unless reviewed ground truth is supplied.
- Model files, caches, downloaded Flickr images, and generated Parquet outputs must not be committed.

The next data-engineering step is to use YOLOE-26 boxes plus BioCLIP object scores to build a reviewed box dataset for later supervised YOLO fine-tuning.
