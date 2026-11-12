# Local EmbeddingGemma 2 benchmark

This experimental workflow compares image content evidence independently of
BioMiner's production GBIF lineage. It does not alter BioCLIP, human reviews,
reference admission, or occurrence release. Raw cosine scores are ranking
evidence, never probabilities. No rejection threshold or automatic deletion is
implemented.

## Install

From the BioMiner repository, create the optional isolated Python 3.14 runtime:

```bash
UV_PROJECT_ENVIRONMENT=.venv-embeddinggemma uv sync --frozen --extra embeddinggemma --extra test
.venv-embeddinggemma/bin/python -m biominer.cli vision embeddinggemma --help
```

The optional extra pins PyTorch 2.15.0, Sentence Transformers 6.3.0, and
Transformers 5.22.2 in `uv.lock`. Other installations need not install the model
stack. The default model is `google/embeddinggemma-2`, pinned to revision
`f9972ac22e696a105c482202646979e33e4b5a1e`. Weights are downloaded to ignored
local storage. The model runs through the official Sentence Transformers
integration with `audio_config=None`, float32 parameters, and no image prompt.
Text labels use the `Classification` prompt.

**Float16 is forbidden.** MPS with float32 is the default; CPU must be selected
explicitly. Disable `PYTORCH_ENABLE_MPS_FALLBACK` so accelerator timings do not
silently include CPU fallback. Bfloat16 is not enabled or evaluated by this
benchmark; any future precision experiment must first reproduce the float32
numerical reference and measure its own performance.

## Validate and run

Set `ORIGINAL_INPUT` and `GEOTAGGED_INPUT` to the two private joined-review
Parquet exports. Set `LABEL_BANK` to the private Markdown vocabulary containing
392 full-sentence labels in 27 groups. Keep these files outside tracked source.

```bash
.venv-embeddinggemma/bin/python -m biominer.cli vision embeddinggemma runtime-check \
  --output-dir data/experiments/embeddinggemma2-v1/runtime-check

.venv-embeddinggemma/bin/python -m biominer.cli vision embeddinggemma run \
  --input "$ORIGINAL_INPUT" --input "$GEOTAGGED_INPUT" \
  --label-bank "$LABEL_BANK" \
  --output-dir data/experiments/embeddinggemma2-v1 \
  --cache-dir data/cache/embeddinggemma2 \
  --cohort all --batch-size 4 --download-workers 4
```

The runtime check measures actual processor visual tokens for both budgets,
checks finite unit vectors, and compares MPS float32 against CPU float32. Its
acceptance limits are cosine at least 0.999 and maximum absolute component
difference at most 0.01. A failed recheck invalidates an earlier passed receipt.
The synthetic image checks numerical plumbing, not biological recognition.

The run proceeds through the 1,500-image balanced cache, the previously
reviewed cohort, and then the complete corpus. `--cohort cache` or
`--cohort reviewed` stops after that cohort. `--limit N` bounds every selected
cohort for an explicit smoke experiment; use a separate output directory for
that experiment. Model numerical checks are still required. Batch size halves
on accelerator out-of-memory errors, stopping with a failed receipt if even
one image cannot be processed.

Each cohort publishes comparisons in this order:

| Order | Maximum visual tokens | Embedding dimensions |
| --- | ---: | ---: |
| 1 | 280 | 256 |
| 2 | 1,120 | 256 |
| 3 | 280 | 768 |
| 4 | 1,120 | 768 |

The processor uses up to its budget; actual token counts depend on image
geometry. Two native 768-dimensional image passes supply all four comparisons.
Compact vectors are the first 256 components, explicitly L2-normalized again.
Both text and image vectors share a dimension. The 256-dimensional benchmark
therefore measures storage and ranking changes, not cheaper encoder inference.
Native vectors remain cached for reuse. Persisted per-configuration vectors
make their storage costs inspectable independently of this shared cache.

## Inputs and interpretation

Preparation preserves every source association, deduplicating image work by
provider identity and then downloaded content hash. Selection uses the latest
round with saved reviews and a strict majority; disputed, uncertain, and
unavailable reviews do not enter the balanced cache. Only explicit reviewed
Papilio target evidence maps target labels into the butterfly group. Search
membership never supplies a label or confirms a species.

The cache has three distinct groups of up to 500 images: non-biological,
other biological, and real adult butterfly/Papilio. The downloaded-byte audit
removes duplicate content and conflicting cross-group content, reporting any
shortfall. Original bytes stay cached; every configuration receives the same
full image with EXIF orientation applied and RGB conversion, without subject
crops. Historical labels remain marked `historical_content_unverified` because
these exports do not prove which exact image bytes the reviewer saw.

The vocabulary remains overlapping discovery evidence. Content top-1/top-5
excludes generic visual-medium labels, which have their own group summary.
Real butterfly presence, depictions, live adults, eggs, larvae, pupae, preserved
specimens, and detached wings retain separate scores. Real preserved adult
specimens also contribute to physical butterfly presence, consistent with
VeriTaxa's broad review label. A flower outranking butterfly-on-flower is
competing evidence, not automatically an error. Existing broad human reviews
cannot establish accuracy for 392 fine labels.

Review queues order weak content evidence, small label margins, close
real-versus-depiction scores, and configuration disagreement. These are ordinal
review priorities without rejection thresholds. The tiny-butterfly gallery
shows model-selected candidates; it does not establish annotated small-subject
accuracy. Balanced-cache results and previously reviewed results are presented
separately and cannot be treated as representative population estimates.

## Artifacts, resume, and individual commands

The experiment output contains:

- `prepared/`: checksummed input manifest, sanitized associations, and items.
- `runtime-check/receipt.json`: numerical/device checks and measured token counts.
- `embeddings/280/` and `embeddings/1120/`: immutable native-vector parts and
  manifest-last checkpoints, reused across cohorts.
- `text/`: checksummed, ordered label embeddings and prompt identity.
- `scores/COHORT/BUDGET-DIMENSION/`: all label scores, content rankings, group
  evidence, normalized image/text vectors, and coverage manifests.
- `comparison/COHORT/index.html`: local report with coverage, exclusions,
  paired changes, group confusion, storage, throughput, memory, and galleries.
- `cache-audit.json`, `media-COHORT.parquet`, and `run.json`: cache audit,
  explicit media failures, PID, current stage, and last completed cohort.

Source URLs and sanitized review evidence stay only in ignored local artifacts.
Reports use local cached thumbnails without external scripts or remote images.
Model weights, source media, caches, and results must never be committed.

Repeat the same `run` command to resume. OS file locks prevent duplicate workers
from writing the same experiment or media cache. Successful cached bytes and
embedding parts are hash-verified before reuse. Changed inputs, prompts,
model/preprocessing settings, or score identity require a new output directory;
do not edit a published manifest to bypass its checks. Corrupt cache bytes are
preserved and excluded for inspection. Failed downloads retain explicit
statuses. A cohort's first media snapshot freezes those statuses and bytes so
all four configurations and resumed work remain comparable. A later cohort
may retry downloads. To revisit failures in an already scored cohort, cache
them with standalone `encode` and write fresh `score` outputs; completed
native embeddings remain reusable through the same embedding directory.

The `prepare`, `encode`, `score`, and `compare` commands can also run separately;
use each command's `--help`. `encode` takes `--prepared-dir`, `--budget`, and a
cache directory. `score` additionally takes `--embedding-dir`, `--label-bank`,
and `--dimension`. `compare` accepts repeated `--configuration-dir` arguments.
Paired comparisons validate source, runtime, label-bank, precision, and cohort
compatibility and require both item identity and identical content hashes.

## Verification

```bash
.venv-embeddinggemma/bin/python -m pytest -q tests/test_embeddinggemma*.py tests/test_cli_dry_run.py
.venv-embeddinggemma/bin/python -m pytest -q
git diff --check
```

Default tests use synthetic media and fake transports/encoders, with no model
weights, live downloads, credentials, or sibling repository fixtures. Live
device checks and corpus runs are opt-in commands with separate receipts.

Primary references: [Google model card](https://huggingface.co/google/embeddinggemma-2)
and [Transformers integration](https://huggingface.co/docs/transformers/model_doc/embedding_gemma2).
