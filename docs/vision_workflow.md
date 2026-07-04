# Vision Workflow

The production vision path is object-first:

```text
canonical Flickr source records
-> YOLOE/YOLO26 object proposals
-> ephemeral crops
-> BioCLIP 2.5 scoring
-> joined object evidence and photo summaries
```

YOLOE/YOLO26 is only an object finder. BioCLIP remains the biological classifier and family, genus, and species scorer.

## Public Commands

The public stage tools are:

```bash
uv run biominer vision detect --help
uv run biominer vision score --help
uv run biominer vision ablate --help
```

Detection writes rows using the stable detection schema with source/photo join keys and object-level IDs. Scoring reads canonical records plus detections and writes BioCLIP object score rows. Ablation compares visual modes over the same input.

Supported visual modes are:

```text
whole_image
detector_crop
detector_crop_segmentation
```

`detector_crop_segmentation` is explicit but only produces segmentation-crop rows when masks are available. The first YOLOE-26 adapter emits boxes only, so regular detector crops are the default useful mode.

There is no image-enhancement mode in production, and BioMiner does not store reviewed boxes or a training dataset as part of this workflow.

## Debug And Runtime Commands

Runtime checks, model prefetch, smoke tests, previews, evaluations, and prototype wrappers live under `biominer dev vision`:

```bash
uv run biominer dev vision yoloe26-runtime-check \
  --runtime-python "../YOLO26/venv/bin/python" \
  --checkpoint yoloe-26s-seg.pt \
  --device auto

uv run biominer dev vision bioclip-runtime-check \
  --runtime-python "../BioCLIP25/venv/bin/python" \
  --device auto

uv run biominer dev vision yoloe26-prototype-run \
  --input staging/species_runs/example/canonical_source_records.parquet \
  --species-context staging/species_runs/example/species_context.json \
  --species-candidates data/registry/current/species_candidates.parquet \
  --output-dir reports/yoloe26_prototype/example \
  --vision-runtime-python "../YOLO26/venv/bin/python" \
  --bioclip-runtime-python "../BioCLIP25/venv/bin/python" \
  --limit 10
```

These commands validate optional runtimes and prototype wiring. They are not the production entry point; production work is coordinated by `biominer run`.

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

Unit tests use fake detectors and fake scorers and must not require Ultralytics, CUDA, MPS, model downloads, or network access.

## Limitations

YOLOE-26 is zero-shot/open-vocabulary. It can miss butterflies and can confuse flowers, leaves, labels, or patterned material with insect-like objects. Low detector confidence can improve recall, but BioCLIP scores and evidence buckets must do the downstream filtering.

YOLOE/YOLO26 output must not be interpreted as species classification. It proposes boxes for BioCLIP; it does not validate family, genus, species, or occurrence identity.
