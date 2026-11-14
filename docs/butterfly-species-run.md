# Local butterfly species ranking

This experimental workflow ranks supplied scientific-name descriptions with
frozen BioCLIP 2.5 Huge. It does not train a model, change human reviews, or
promote images into BioMiner's production reference or occurrence pipeline.

## Frozen image cohort

Use the saved EmbeddingGemma benchmark snapshot, without refreshing live
VeriTaxa reviews. Include the union of decisive human `adult_butterfly` or
qualified `target_scientific_name:Papilio demoleus` decisions and strictly
unreviewed images whose top content group is butterfly in all four configurations.
Human rejection, dispute, and unavailable states cannot enter through the
unreviewed branch. Human adult decisions do not require model agreement.

The selected experiment has 12,058 image identities: 7,102 human-labelled adults
and 4,956 unreviewed four-way matches. All original/geotagged source associations
are preserved. The selection is tied to checksummed source outputs and local
image bytes. Historical human reviews remain unverified against those bytes.

## Prepare

From BioMiner, use the existing Python 3.14 administration environment with
Polars, NumPy, Pillow, and PyArrow. The isolated EmbeddingGemma environment
installed by the preceding benchmark supplies these packages.

```bash
.venv-embeddinggemma/bin/python -m biominer.vision.species_selection \
  --benchmark-dir data/experiments/embeddinggemma2-v1 \
  --cache-dir data/cache/embeddinggemma2 \
  --names "$SPECIES_NAMES_FILE" \
  --target-scientific-name "$TARGET_SCIENTIFIC_NAME" \
  --output-dir "$SPECIES_SELECTION_DIR"
```

The names file accepts comma- or newline-separated descriptions. Exact
duplicates use their first occurrence; spelling, case, synonyms, and order are
preserved. No taxonomic authority reconciliation or extra candidate labels are
performed. The selection contains `selected_images.parquet`, all associations,
`labels.json`, `label_audit.json`, and a manifest published last. Changed source
fingerprints or edited artifacts fail closed and require a new output directory.

The supplied experiment uses 250 requested entries, 244 unique descriptions,
and six exact duplicate entries. Its mode is `species`, with one `species`
head. The existing subject/context mode remains available for other runs.

## Run

Use BioMiner's administration Python to coordinate the separate, existing
BioCLIP model runtime. Set the selection and output directories explicitly:

```bash
.venv-embeddinggemma/bin/python -m biominer.vision.custom_label_run \
  --source "$SPECIES_SELECTION_DIR/selected_images.parquet" \
  --labels "$SPECIES_SELECTION_DIR/labels.json" \
  --output "$SPECIES_RUN_DIR" \
  --model-home "$BIOCLIP_MODEL_HOME" \
  --model-python "$BIOCLIP_MODEL_PYTHON" \
  --hf-cache "$BIOCLIP_HF_CACHE" \
  --device mps --batch-size 8 --download-workers 4
```

The model is `imageomics/bioclip-2.5-vith14`, pinned to
`191d741545e4c741cdef4b22c6eb69c945c1e592`, using OpenCLIP 3.3.0's default
float32 precision and no autocast. The run attests weights, model config, and
preprocessing. Disable `PYTORCH_ENABLE_MPS_FALLBACK` for accelerator measurements.
No YOLOE stage is invoked. No sibling repository source is modified.

Rows with `cached_image_path` must include a matching `cached_image_sha256`.
They use only those local bytes, with no network fallback. Missing, corrupt,
oversized, or undecodable local images receive explicit failure evidence.
Canonical PNGs retain the complete frame with EXIF orientation applied and RGB
conversion. The model uses BioMiner's attested longest-side preprocessing.

Each scientific name uses three photographic templates: a photo, a close-up
photo, and a photo of the butterfly. Text embeddings are normalized, averaged
per name, and normalized again. Images are encoded once independently of label
count. Optional repeated `--cached-bioclip-state PATH` arguments open previous
SQLite states read-only and reuse vectors only through exact content/model/
preprocessing keys, with finite-vector, dimension, and unit-norm checks.
EmbeddingGemma embeddings are never used as BioCLIP embeddings.

A deterministic 32-image smoke stage balances human/unreviewed selection reasons
and collection strata. It validates image-content binding and embeddings,
records memory, throughput, and an ETA, then automatically proceeds. Memory
exhaustion reduces the batch size; irrecoverable inference errors stop the job
and retain its checkpoint. An OS lock prevents two workers writing the same run.
Repeat the exact command to resume an unchanged input/configuration.

## Outputs and interpretation

`current_results.json` points to the active scoring-fingerprint directory.
That directory contains checkpointed `image_suggestions.parquet`, `progress.json`,
`smoke_test.json`, model/prompt attestations, and final `all_label_scores.parquet`
and `report.html`. The SQLite work state preserves native image embeddings
separately from predictions, allowing new label sets to reuse image encodings.

New species rows contain top-ten scientific names and their cosine scores, raw cosine scores for every
candidate, and the top-two cosine margin. All selected records remain represented,
including pending and failed records. Missing evidence stays null. Finalization
is separate from all records merely having a terminal status.

The report compares Papilio demoleus top-one/top-five/top-ten suggestions against the
saved qualified human decisions. Generic adult-butterfly decisions cannot
establish species accuracy. Rankings are conditional on the supplied list;
omitted species cannot be returned. Cosines and margins are not calibrated
probabilities. Species identity and wild-occurrence status remain unresolved.

Top-ten outputs use evidence schema v2 and include the saved rank count in the
species scoring fingerprint. This prevents old top-five prediction payloads
from being reused as complete top-ten results. Compatible native image
embeddings retain their existing cache keys. Pending/failed rows and runs with
fewer than ten candidates retain typed null rank fields.

All runtime files, original media, private source metadata, model weights, and
reports belong in ignored local storage and must not be committed or published.

## Verification

```bash
.venv-embeddinggemma/bin/python -m pytest -q \
  tests/test_species_selection.py tests/test_custom_label_species.py \
  tests/test_custom_label_run.py
git diff --check
```

Default tests use synthetic image bytes and fake scorers and make no live calls.
The actual device smoke stage is recorded separately from those test results.

Integration evidence: [PyBioCLIP custom-label normalization](https://github.com/imageomics/pybioclip/blob/256e467401a57c5e2cd8acb1467fbd834d757e9d/src/bioclip/predict.py)
and the [pinned BioCLIP 2.5 Huge model card](https://huggingface.co/imageomics/bioclip-2.5-vith14/blob/191d741545e4c741cdef4b22c6eb69c945c1e592/README.md).
