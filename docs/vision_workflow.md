# Vision Workflow

The production vision path is object-first:

```text
canonical Flickr source records
-> YOLOE/YOLO26 object proposals
-> ephemeral crops
-> BioCLIP 2.5 scoring
-> joined object evidence and photo summaries
```

YOLOE/YOLO26 is only an object finder. Production sends only `butterfly_like` detections with `detection_status=detected` to BioCLIP; moth, caterpillar, pupa, generic insect, hard-negative, no-detection, and failed-image rows remain evidence but are not species-scored. Production also avoids creating non-debug crop artifacts for those non-eligible detections.

The current default classification mode is `target_scope_object_screening`. BioCLIP scores detector crops against target/scope candidate labels for screening evidence. Existing columns named `family_top3`, `species_top20`, and `species_top5` are not yet a true family-first hierarchical classifier: `species_top20` is not constrained by `family_top1`, and the current top-5 list is recorded with the target-screening rerank strategy and top-k settings used for that row.

The reserved `hierarchical_butterfly_classification` mode is for a later GBIF taxonomy candidate-table workflow. It can be recorded in dry-run plans, but real scoring fails clearly until that classifier exists.

The production default visual mode is `detector_crop`. Whole-image BioCLIP is available only through explicit ablation/debug commands because it spends model budget on background, host plants, labels, hands, and other non-target content.

## Public Commands

The public stage tools are:

```bash
uv run biominer vision screen --help
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

YOLO26 checkpoints must emit BioMiner coarse object labels or known legacy object aliases. Species-class checkpoints are rejected rather than remapped.

For a local detector-first run that keeps each stage as durable zstd part files, use `vision screen`. It runs one persistent YOLOE sidecar and one persistent BioCLIP sidecar, writes canonical/detection/score/joined/summary part directories, and deletes cached images only after the relevant outputs commit.

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
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer dev vision yoloe26-runtime-check \
  --runtime-python "../YOLO26/venv/bin/python" \
  --checkpoint yoloe-26s-seg.pt \
  --device mps

PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer dev vision bioclip-runtime-check \
  --runtime-python "../BioCLIP25/venv/bin/python" \
  --device mps

uv run biominer dev vision yoloe26-prototype-run \
  --input runs/local_debug/papilio_demoleus/canonical_source_records.parquet \
  --species-context runs/local_debug/papilio_demoleus/species_context.json \
  --species-candidates data/registry/current/species_candidates.parquet \
  --output-dir reports/yoloe26_prototype/example \
  --vision-profile mac_m5pro_64gb \
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

The Mac M5 Pro / 64 GB profile is `mac_m5pro_64gb`. It uses Apple MPS, YOLOE checkpoint `yoloe-26s-seg.pt`, YOLO image size `768`, detector batch size `16`, crop batch size `24`, crop target `336`, crop padding `0.08`, zstd Parquet part outputs, and delete-after-commit image cleanup. Use `PYTORCH_ENABLE_MPS_FALLBACK=1` for runtime checks and sidecar runs.

Unit tests use fake detectors and fake scorers and must not require Ultralytics, CUDA, MPS, model downloads, or network access.

## Limitations

YOLOE-26 is zero-shot/open-vocabulary. It can miss butterflies and can confuse flowers, leaves, labels, or patterned material with insect-like objects. Low detector confidence can improve recall, but BioCLIP scores and evidence buckets must do the downstream filtering.

YOLOE/YOLO26 output must not be interpreted as species classification. It proposes boxes for BioCLIP; it does not validate family, genus, species, or occurrence identity.
