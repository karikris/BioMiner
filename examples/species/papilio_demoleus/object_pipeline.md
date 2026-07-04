# Papilio demoleus Object Pipeline Example

This example keeps `Papilio demoleus` as fixture data and a CLI input only. The runnable path is the generic single-species object pipeline.

Resolve the species scope through the production run planner:

```bash
uv run biominer run \
  --taxon "Papilio demoleus" \
  --rank species \
  --registry-dir data/registry/current \
  --output-prefix runs/local_debug/papilio_demoleus \
  --storage-backend local \
  --workstore-backend sqlite \
  --stages resolve \
  --dry-run
```

Run object detection on the filtered canonical records:

```bash
uv run biominer vision detect \
  --input runs/local_debug/papilio_demoleus/filtered.parquet \
  --output runs/local_debug/papilio_demoleus/object_detections.parquet \
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
  --species-context runs/local_debug/papilio_demoleus/species_context.json \
  --input runs/local_debug/papilio_demoleus/filtered.parquet \
  --detections runs/local_debug/papilio_demoleus/object_detections.parquet \
  --species-candidates data/registry/current/species_candidates.parquet \
  --output runs/local_debug/papilio_demoleus/object_bioclip_scores.parquet \
  --ablation-mode detector_crop \
  --device auto \
  --parquet-batch-rows 10000
```

Run the whole-image, detector-crop, and crop-plus-segmentation ablations:

```bash
uv run biominer vision ablate \
  --species-context runs/local_debug/papilio_demoleus/species_context.json \
  --input runs/local_debug/papilio_demoleus/filtered.parquet \
  --detections runs/local_debug/papilio_demoleus/object_detections.parquet \
  --species-candidates data/registry/current/species_candidates.parquet \
  --output-dir runs/local_debug/papilio_demoleus/ablations \
  --modes whole_image,detector_crop,detector_crop_segmentation \
  --device auto \
  --parquet-batch-rows 10000
```

Join canonical records, object detections, and object BioCLIP scores:

```bash
uv run biominer evidence join \
  --species-context runs/local_debug/papilio_demoleus/species_context.json \
  --input runs/local_debug/papilio_demoleus/filtered.parquet \
  --detections runs/local_debug/papilio_demoleus/object_detections.parquet \
  --scores runs/local_debug/papilio_demoleus/object_bioclip_scores.parquet \
  --joined-output runs/local_debug/papilio_demoleus/object_evidence_joined.parquet \
  --photo-summary-output runs/local_debug/papilio_demoleus/photo_evidence_summary.parquet
```

The same commands work for another species by changing the `--scientific-name`, `--species-context`, and run directory paths.
