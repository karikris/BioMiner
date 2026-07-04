# Papilio demoleus Object Pipeline Example

This example keeps `Papilio demoleus` as fixture data and a CLI input only. The runnable path is the generic single-species object pipeline.

Resolve the species context first:

```bash
uv run biominer species resolve \
  --scientific-name "Papilio demoleus" \
  --registry-dir data/registry/current \
  --output-root staging/species_runs/papilio_demoleus
```

Run object detection on the filtered canonical records:

```bash
uv run biominer vision detect \
  --input staging/species_runs/papilio_demoleus/filtered.parquet \
  --output staging/species_runs/papilio_demoleus/object_detections.parquet \
  --backend yoloe26 \
  --runtime-python ./YOLO26/venv/bin/python \
  --checkpoint yoloe-26s-seg.pt \
  --device auto \
  --image-max-side-px 1280 \
  --detector-batch-size 4 \
  --max-inflight-images 32 \
  --max-inflight-crops 96
```

Score detector crops with BioCLIP against registry-derived candidate labels:

```bash
uv run biominer vision score \
  --species-context staging/species_runs/papilio_demoleus/species_context.json \
  --input staging/species_runs/papilio_demoleus/filtered.parquet \
  --detections staging/species_runs/papilio_demoleus/object_detections.parquet \
  --species-candidates data/registry/current/species_candidates.parquet \
  --output staging/species_runs/papilio_demoleus/object_bioclip_scores.parquet \
  --ablation-mode detector_crop \
  --device auto \
  --parquet-batch-rows 10000
```

Run the whole-image, detector-crop, and crop-plus-segmentation ablations:

```bash
uv run biominer vision ablate \
  --species-context staging/species_runs/papilio_demoleus/species_context.json \
  --input staging/species_runs/papilio_demoleus/filtered.parquet \
  --detections staging/species_runs/papilio_demoleus/object_detections.parquet \
  --species-candidates data/registry/current/species_candidates.parquet \
  --output-dir staging/species_runs/papilio_demoleus/ablations \
  --modes whole_image,detector_crop,detector_crop_segmentation \
  --device auto \
  --parquet-batch-rows 10000
```

Join canonical records, object detections, and object BioCLIP scores:

```bash
uv run biominer vision join \
  --species-context staging/species_runs/papilio_demoleus/species_context.json \
  --input staging/species_runs/papilio_demoleus/filtered.parquet \
  --detections staging/species_runs/papilio_demoleus/object_detections.parquet \
  --scores staging/species_runs/papilio_demoleus/object_bioclip_scores.parquet \
  --joined-output staging/species_runs/papilio_demoleus/object_evidence_joined.parquet \
  --photo-summary-output staging/species_runs/papilio_demoleus/photo_evidence_summary.parquet
```

The same commands work for another species by changing the `--scientific-name`, `--species-context`, and run directory paths.
