# YOLOE26 replacement with EmbeddingGemma 2

Task: `embeddinggemma2-routing-2026-11-17`.
Starting branch/commit: `main`, `3457d07feeb0526e074ae08e14d30a891c1dfb11`.

## Analysis and resulting behavior

BioMiner used YOLOE26 to propose objects, localize them, assign coarse visual
domains, and aggregate reference quality evidence. BioCLIP supplied independent
taxonomic ranking. Replacing the first model changes the available evidence:
EmbeddingGemma 2 produces image/text vectors and similarities, without object
localization or measured subject size. A model-name substitution would have
misrepresented those missing capabilities.

The replacement uses frozen `google/embeddinggemma-2`, revision
`f9972ac22e696a105c482202646979e33e4b5a1e`. The revision was retrieved from the
official Hugging Face model API on 2026-11-17. Google documents native multimodal
embeddings with 768 dimensions; the text/vision configuration omits the audio
encoder. The adapter uses float32 by default and permits bfloat16, never
float16. It requires Sentence Transformers >=6.1 and Transformers >=5.18.

Primary evidence:

- [Google multimodal usage](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers).
- [Google developer guide](https://developers.googleblog.com/en/embeddinggemma-2-the-developer-guide/).
- [Official model card](https://huggingface.co/google/embeddinggemma-2).
- [Pinned model configuration](https://huggingface.co/google/embeddinggemma-2/blob/f9972ac22e696a105c482202646979e33e4b5a1e/config.json).
- [Pinned task prompts and producing library versions](https://huggingface.co/google/embeddinggemma-2/blob/f9972ac22e696a105c482202646979e33e4b5a1e/config_sentence_transformers.json).

## Workflow inventory

| Previous component | Replacement and boundary |
| --- | --- |
| `detection/yoloe26_detector.py`, direct model and persistent YOLOE sidecar | `detection/embeddinggemma2.py` and `embeddinggemma2_worker.py`: frozen full RGB frame, one 768D vector per image, fixed complete prompt universe, cached text vectors. |
| YOLOE runtime check/prefetch/smoke commands | `dev vision embeddinggemma2-runtime-check`, `embeddinggemma2-prefetch`, `embeddinggemma2-smoke`. Runtime check does not load weights. |
| `yoloe-flickr-sample` and `yoloe-flickr-sharded` | `embedding-flickr-sample` and `embedding-flickr-sharded`, using `flickr_fetch/embedding_screening.py`. Old execution commands are rejected. |
| YOLO26 runtime directory/setup | Dedicated `EmbeddingGemma2` runtime/cache, shared by prefetch and Flickr; `scripts/setup_embeddinggemma2.sh` and optional `embeddinggemma2` package extra. |
| Orchestrator declared backend and incremental detection cache | Backend `embeddinggemma2`, cache scope `embeddinggemma2_routing`, `run_visual_routing`/`reuse_visual_routing` actions. Old YOLOE caches cannot match the new scope. |
| Reference object-detection aggregation | `references/embedding_routing.py`, separate source-bound review schema and raw-frame identity. No automatic admission from embedding-only evidence. |
| Geometry-dependent BioCLIP planning | `build_embedding_routed_full_frame_plan`: distinct image route evidence, the same raw canvas, independent frozen BioCLIP preprocessing and encoding. Spatial attention generation is rejected without localization. |
| Flickr scoring-unit artifacts | `flickr-scoring-unit-v2.0.0`, explicit `full_frame_embedding_route` versus `detected_object_route`. Image routes retain review status and null detection IDs/counts; associations and candidate unions are retained. |
| Historical detection payload/prompt identity | Pure read-only codecs moved to `detection/legacy_yoloe_evidence.py`. Historical measured-detection/reference/ML/review schemas remain distinct. |
| `scripts/run_local_rolling_vision.py` | Removed: it already depended on absent rolling-worker and crop modules. Its obsolete crop workflow was not revived. |

The production orchestrator remains governed stage infrastructure: live model
stages require explicit handlers and validated upstream manifests. Updating its
backend declaration does not execute or claim completion of a production job.

## Evidence and compatibility contracts

Every result retains all prompt similarities, the highest similarity, and the
margin against the best different domain. A complete fixed prompt universe
keeps adult butterfly, moth/other insect, larva, pupa, specimen, artifact, and
no-organism alternatives. Image input includes no source query text or metadata.
Default similarity/margin cutoffs are provisional heuristics; **all outcomes
require review**, including apparently clear or negative proposals.

Cosine values are signed raw evidence in [-1, 1], never detector confidences or
probabilities. No-organism is a proposed domain, not demonstrated biological
absence. Subject presence, subject area, detection counts, objectness, bounding
boxes, and masks remain unavailable. References cannot satisfy the historical
geometry-dependent automatic admission compiler from these outputs. Human
review, independent QA, rights, and occurrence release remain separate gates.

The Flickr scoring table retains `organism_unit_id` as its historical foreign-key
spelling. For the new schema its `scoring_unit_kind` identifies an image/route
unit; it must not be counted as an observed organism. New image units carry no
measured detection evidence. Existing YOLOE-named ML/export fields are historical
contracts and are not silently filled from Gemma proposals.

The Flickr workflow snapshots source/photo/query provenance into checksummed
Parquet using bounded SQLite/Parquet iteration. Resume binds source snapshot,
requested config, immutable model revision, actual worker runtime, full-frame
preprocessing, prompt universe, score policy, and schema. Legacy/unbound state,
changed snapshots, model/runtime changes, and concurrent writers fail closed.
Each shard owns one persistent model at a time. A model-wide failure replaces
that worker within a bounded restart budget; exhaustion aborts with checkpoints
intact. It cannot turn untouched images into terminal biological negatives.
The coordinator writes completion manifests after durable outputs and reports.

## Usage

Install the optional model runtime using Python >=3.14.7:

```bash
PYTHON_BIN=/path/to/python3.14 bash scripts/setup_embeddinggemma2.sh
biominer dev vision embeddinggemma2-runtime-check
biominer dev vision embeddinggemma2-prefetch
```

Run a bounded sample only after producing an authorized post-ground-zero source
snapshot, with a fresh output directory:

```bash
biominer dev vision embedding-flickr-sample \
  --state-db /path/to/new-source-snapshot.sqlite \
  --output-dir /path/to/new-embedding-output \
  --sample-size 100 --batch-size 4
```

For a complete immutable snapshot, use `embedding-flickr-sharded`, `--shards`,
and optionally `--expected-images`. `--runtime-python` selects an explicit
interpreter. Existing HF cache overrides are preserved. System interpreters use
the configured runtime cache; venv interpreters may use their own runtime root.
The `embeddinggemma2-smoke` command requires a supplied `--image` and an empty
output directory. None of these model/discovery commands was run for this task.

## Validation and limitations

The available native test interpreter was
`.native-runtime/application-p2a26-20261022-1/env/bin/python` (Python 3.14.7).
The system interpreter lacked Pillow/httpx and was not used as validation
evidence. The initial focused baseline passed 81 tests.

The focused replacement/integration matrix passed 149 tests, including model
vectors, lazy imports, frozen loader options, complete prompt routing, signed
scores, worker protocol/timeouts, immutable resume, shard isolation, runtime
cache sharing, source-bound references, and independent raw-frame BioCLIP.
After review fixes, the final focused command across 22 affected and neighboring
test modules passed **351 tests in 7.90 seconds**. This includes the two system
interpreter cache regressions and legacy evidence/readiness boundaries.

The first full regression run passed 4,478 tests and failed one pre-existing
check: `tests/test_registry_species_context.py::test_production_modules_do_not_hardcode_papilio_species`.
It flags the source label `Papilio demoleus multilingual candidate table` in
`registry/image_keyword_enrichment.py:35`, already present at the starting
commit. That label and the unrelated registry module were preserved.

The final full regression command, `python -m pytest -q` with the native
interpreter above, passed **4,487 tests** and failed the same one pre-existing
check in 182.80 seconds. The later two cache-path regressions passed in the
final focused run. Full output is retained at
`/tmp/biominer-embeddinggemma2-final.8UHqcj.log` for this workspace session.

`uv lock --check`, focused Ruff undefined/unused-name checks, setup-script syntax,
and `git diff --check` passed. The optional runtime dependencies were resolved
into `uv.lock`; no model weights were downloaded and no biological accuracy,
throughput, GPU memory, or release-readiness claim is supported. Existing
immutable native-runtime receipts were not rewritten or activated; a fresh
qualified generation is required before a production native-runtime cutover.

Independent backend specification review and integration code review were
performed. Review findings led to dependency-version checks, strict JSON and
request identities, bounded pipe writes/timeouts, worker restart handling,
shared cache paths, domain mapping, and explicit image-level scoring contracts.

## Ownership, provenance, and stop conditions

The initial worktree contained active GBIF reporting/scripts/tests and a dirty
`AGENTS.md`; none was staged, committed, or altered by this task. The instruction
to commit dirty main first conflicted with the prohibition on committing another
session's work. The resolution was to preserve it and commit only reviewed task
paths on main. The user's current replacement request supersedes the old YOLOE
direction in that dirty instruction file; current pipeline docs were updated.

GitHits source precedent came from an older indexed Sentence Transformers
snapshot (`688fd663`); native multimodal `encode` plus explicit task prompting
was adopted. Task-asymmetric query/document encoding and hand-written low-level
pooling were not needed. No external source code was copied. Valyu discovery
was checked against official Google documentation and the pinned model files.
The task record is appended to `provenance/githits.jsonl`.

Historical `config/pilot` examples describe retired staging workflows and are
not runnable production configuration for the new commands. Measured-detection
IoU evaluation and YOLOE-named ML features remain historical contracts; they do
not establish evaluation results for embeddings.

No previous staging registry, corpus, model output, report, or handoff was
resumed or used as current production evidence. The raw GBIF DWCA remains the
only production lineage. Non-fast-forward push, active-file overlap, incompatible
state, and missing evidence needed for admission are stop conditions. Rollback
is a later task-specific revert, never a reset of the shared worktree.
