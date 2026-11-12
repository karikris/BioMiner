# Flickr full-frame custom-label experiment

This is the user's explicitly requested historical Flickr experiment, outside
the GBIF ground-zero production lineage. It does not change production intake,
the live Veritaxa website, saved human reviews, or occurrence-release eligibility.

## Contract

`scripts/run_flickr_custom_labels.py` runs frozen BioCLIP 2.5 Huge, pinned to
`imageomics/bioclip-2.5-vith14` revision
`191d741545e4c741cdef4b22c6eb69c945c1e592`. The existing persistent BioMiner
worker attests model weights, OpenCLIP configuration and preprocessing. The
runtime must support the pinned model; this command needs no Flickr API key.

Every source photo receives a separate binding, with exactly one source row per
Flickr photo ID. Supplied large Flickr renditions are preferred, with a supplied
medium rendition as fallback. URLs are never constructed by changing suffixes.
Downloads are bounded to 20 MiB, 50 million decoded pixels, three attempts per
rendition and a 30-second socket timeout. Four download threads feed batches of
eight images. EXIF orientation is applied, the complete RGB frame is cached,
and the model uses longest-side resize with centered padding at 224 pixels.
There is no detector, crop, hierarchy pruning or metadata-driven prediction.

The worker starts once and retains the model. MPS memory errors halve the image
batch; failure at batch size one stops with resumable state. SQLite checkpoints
retain media provenance, decoded-content embeddings and per-prompt predictions.
Embeddings are keyed by decoded RGB content, media preparation, attested model
identity and preprocessing. Prompt changes rescore existing embeddings. Failed
downloads have explicit failure rows and no invented vectors or labels.

Each description is encoded under two photographic templates. Unit text vectors
are averaged per category and renormalized, following the pattern in
[pybioclip's CustomLabelsClassifier](https://github.com/Imageomics/pybioclip/blob/256e4674/src/bioclip/predict.py#L351).
Subject and context heads rank their categories independently using raw cosine
similarity. All scores, top three labels and first/second margins are retained.
No score is a calibrated probability. Wild status is always unresolved.

## Run or resume

Install the lightweight controller extra with the project's normal environment
manager, for example `uv sync --extra classification --extra test`. Keep Torch
and OpenCLIP in the separate model runtime already used by BioMiner.

```bash
.venv/bin/python scripts/run_flickr_custom_labels.py \
  --source /path/to/geotagged_source_records.parquet \
  --metadata /path/to/enriched_images.parquet \
  --discovery-state /path/to/flickr_poller.sqlite \
  --labels config/vision/flickr_custom_labels_v3.json \
  --output /path/to/local-run-directory \
  --model-home /path/to/BioCLIP25 \
  --model-python /path/to/BioCLIP25/venv/bin/python \
  --hf-cache /path/to/BioCLIP25/cache/huggingface \
  --device mps --batch-size 8 --download-workers 4
```

The first invocation freezes source and optional enriched metadata snapshots.
Metadata must join 1:1 by photo ID and match source-record hashes. Overlapping
columns use the enriched snapshot's values, and source-only columns are retained.
Human-review columns remain source evidence; model outputs use `bioclip_` names.
IDs, source URLs and metadata are never sent to the text encoder.

Re-run the same command to resume. `--retry-failures` retries terminal media
failures; successful predictions remain cached. A changed prompt file creates a
new results directory in the same run root and reuses compatible embeddings.
A changed input source requires a new output root. To incorporate newer reviews
or metadata, use a new output root and its explicit new metadata snapshot; an
existing run never silently replaces its frozen metadata.

The run directory and all media, metadata, caches and result files are local
ignored runtime artifacts, and must never be staged in Git. A nonblocking file
lock prevents concurrent writers to the same run root. Interrupting the process
retains completed inference batches; resume writes current exports from state.

## Outputs

`current_results.json` identifies the active scoring fingerprint's directory.
That directory contains:

- `image_suggestions.csv` and `.parquet`: every source row, including failures
  and pending rows during execution, enriched metadata and AI-only columns.
- `all_label_scores.parquet`: one row per scored image, head and category.
- `image_suggestions.xlsx`: all suggestions, each nonempty top-subject group,
  explicit unscored rows if present, and a review-sample tab. Long text is split
  losslessly into consecutive `__part_N` columns at Excel's UTF-16 cell limit.
  Strings are literal values, never formulas or implicit hyperlinks.
- `review_sample_200.csv` and `review_sample_design.json`: 100 random and 100
  targeted-risk images, with blank manual-review columns. Targeted selection
  prioritizes questionable coordinates, nonordinary context, metadata risk flags
  and small subject margins.
  The deterministic first 32 development images are excluded from this audit.
  Only random-stratum reviews can estimate performance in the eligible scored
  population; targeted rows cannot support an unweighted accuracy claim.
- `run_manifest.json`, `prompts.json`, `text_prototypes.npy`, `smoke_test.json`
  and `progress.json`: pinned inputs, model identity, actual preprocessing,
  prompt configuration, counts, timing, memory and final artifact fingerprints.

CSV/Parquet checkpoints are replaced atomically every 100 completed bindings.
The final workbook and score table are written after inference; require
`progress.json.artifacts_finalized=true` before consuming final exports.

## Prompt-development findings and limits

Version 1 used negative descriptions such as "an insect other than a butterfly
or moth". Inspection of the 32-image development set found butterflies routed to
that overlapping category. Version 2 replaced negatives with positive examples,
but a "butterfly chrysalis" description still attracted adult butterfly images.
Version 3 uses short vernacular stage descriptions and preserves all scores.
Both earlier runs remain in separate fingerprinted directories for audit.

Spot checks also found incorrect stage and context suggestions. These are
unvalidated model rankings, not reliable butterfly counts, life-stage labels,
wild/captive decisions or occurrence records. No species identification or
automatic exclusion is performed. The final 200-image sample still requires
human review; no biological accuracy has been measured. Do not promote these
rankings into saved human classifications or filter the source corpus with them.

All 3,638 source photos were scored with zero media failures. Final integrity
checks found three `(0, 0)` coordinate pairs already present in the original
Flickr search data, each with provider accuracy level 16. These remain in the
dataset and are explicitly flagged `zero_zero_requires_review` in
`bioclip_input_geolocation_check`; they are included in the review sample.
Other coordinates are labelled `present_unverified`, rather than verified
locations. Adding this export check reused the completed predictions; the
69,122-row all-label score table remained byte-for-byte unchanged.

The initial 32-image run produced 32 valid embeddings in 5.85 seconds, with
approximately 4.48 GB peak MPS driver allocation. Subsequent prompt runs reused
the compatible image embeddings. Actual final counts and timings belong to the
local checksum-bound progress artifact, not an extrapolated benchmark.

## Verification and ownership

Starting branch: `agent/yoloe-flickr-pilot`, starting commit
`9a26a069e2cb464526215f58452dd92a534ed1b8`. Pre-existing YOLOE edits and local
archives/logs were left untouched. No source data or model artifacts belong in
the implementation commit.

Focused tests exercise normalization, raw score semantics, multiple positive
descriptions, full panorama preservation, EXIF orientation, invalid/oversized
downloads, supplied URL fallback, checksum cache reuse, source-bound metadata
joins, prompt-only rescoring, duplicate-content reuse, memory backoff and
lossless Excel text. They use synthetic images and fake scorers only.

Commands used:

```bash
.venv/bin/python -m pytest -q tests/test_custom_label_run.py
.venv/bin/python -m pytest -q tests/test_custom_label_run.py tests/test_bioclip_worker.py tests/test_bioclip_worker_batching.py tests/test_full_frame_attention.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_flickr_custom_labels.py --help
git diff --check
```

Final focused tests: 20 cases passed. The focused tests together with neighboring
BioCLIP/full-frame worker tests: 87 passed. Final full working-tree regression:
2,642 passed in 95.23 seconds. This includes pre-existing unstaged YOLOE work,
which is excluded from the custom-label implementation commit. CLI help and
offline dependency-lock resolution also passed. Real artifact verification
confirmed all 3,638 IDs exactly once, exact preservation of the enriched
metadata, exact CSV/Parquet round trip, all workbook row values (numeric
tolerance `1e-12`), 69,122 unique finite label scores, 3,622 unique normalized
image embeddings and the 100/100 review sample including all three coordinate
flags. Final artifact hashes and checks belong to the local run receipt.
