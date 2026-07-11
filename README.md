# BioMiner

BioMiner builds a reviewed butterfly taxonomy registry, discovers Flickr records, removes non-butterfly material, detects and crops butterflies, and screens each crop with BioCLIP 2.5. GBIF defines accepted species identity. Classifier output is evidence for review, never taxonomic authority.

## Architecture

The production command is `biominer run`. It executes a resumable staged pipeline:

1. build or validate the Step 0 registry;
2. compile atomic Flickr query definitions;
3. fetch metadata only;
4. filter metadata and run YOLOE detection;
5. classify eligible crops through `FAMILY → SUBFAMILY → TRIBE → SUBTRIBE → GENUS → SPECIES`;
6. join evidence, assign review buckets, and optionally evaluate Flickr comments.

The rolling vision worker is the only production visual path. It uses bounded queues, immutable Parquet parts, commit-ordered cleanup, and persistent model workers. Direct detect, screen, score, rolling-screen, and ablation commands have been removed.

Durable tabular artifacts are Parquet. DuckDB is used for local audits and summaries. Local work state uses SQLite; cloud runs use PostgreSQL-compatible work storage and S3-compatible object storage.

## Taxonomy contract

The base registry contains accepted GBIF taxa and names. The classifier consumes a separate reviewed overlay:

```text
Family:    Papilionidae
Subfamily: Papilioninae
Tribe:     Papilionini
Subtribe:  [reviewed skip]
Genus:     Papilio
Species:   Papilio demoleus
GBIF key:  1938069
```

Every enabled node, edge, and species mapping records authority, release, citation, retrieval date, evidence, reviewer, and review date. A missing SUBTRIBE is allowed only as an explicit reviewed `TRIBE → GENUS` skip; BioMiner does not invent a subtribe. Missing, conflicting, or unreviewed paths remain disabled. The current curated seed is intentionally narrow; unmapped accepted GBIF species are emitted as explicit QA gaps.

Classification-v3 artifacts:

```text
classification_sources.parquet
classification_nodes.parquet
classification_edges.parquet
species_gbif_mappings.parquet
classification_leaf_paths.parquet
classification_prompt_labels.parquet
classification_qa_findings.parquet
classification_manifest.json
```

Production uses one fixed `global_rank_top_k` beam of 3. At each rank from
FAMILY through GENUS, it scores the deduplicated union of actual nodes in the
surviving paths and keeps the global three by current-rank raw similarity. The
three may share one parent; there is no per-parent quota or cumulative path
score. Reviewed SUBTRIBE skips pass through without consuming a beam position.
The surviving genus paths define the species universe, followed by a fixed
species top 20, distinct-prompt rerank top 5, and reported top 3.

Classification output separates rank-local `<rank>_top1` fields from
`selected_<rank>` fields on the final species-winning path. It also persists
`classification_fingerprint`, `hierarchy_fingerprint`, and
`embedding_cache_fingerprint`. Hierarchical production requires a complete
classification-v3 cache built for the active BioCLIP model and checkpoint;
missing or mismatched caches fail closed.

## Installation

```bash
uv sync
uv run biominer --help
```

Secrets belong in environment variables or an uncommitted `.env`. Do not commit keys, raw API dumps, downloaded images, model weights, caches, or generated registry builds.

## Quick start

Build the accepted GBIF registry:

```bash
uv run biominer registry build \
  --output-dir data/registry/butterflies-v2 \
  --registry-version butterflies-v2 \
  --workers 8 \
  --progress-every 100 \
  --checkpoint-every 500 \
  --max-retries 5
```

Compile the reviewed six-rank overlay (SUBTRIBE may be an explicit reviewed skip):

```bash
uv run biominer registry build-classification \
  --registry-dir data/registry/butterflies-v2 \
  --source-json config/taxonomy/papilionoidea_classification_v3.json
```

Precompute its fingerprinted BioCLIP text embeddings once per taxonomy,
prompt, model, or checkpoint version:

```bash
uv run biominer dev vision build-text-embedding-cache \
  --taxonomy-candidate-table data/registry/butterflies-v2 \
  --output data/registry/butterflies-v2/classification_text_embeddings.parquet
```

Audit the registry:

```bash
uv run biominer registry audit \
  --registry-dir data/registry/butterflies-v2 \
  --report-dir reports
```

Inspect a production run without executing stages:

```bash
uv run biominer --config config/biominer.local.example.toml run \
  --taxon Papilionidae \
  --rank family \
  --registry-dir data/registry/butterflies-v2 \
  --taxonomy-candidate-table data/registry/butterflies-v2 \
  --taxonomy-text-embedding-cache data/registry/butterflies-v2/classification_text_embeddings.parquet \
  --output-prefix data/runs \
  --storage-backend local \
  --workstore-backend sqlite \
  --classification-mode hierarchical_butterfly_classification \
  --dry-run
```

`--dry-run` resolves the scope and emits the run plan without executing stages;
it does not load or validate classification-v3 artifacts or the embedding cache.
Those inputs fail closed when the hierarchical vision stage initializes.

## Supported command surface

- `biominer run`: sole production workflow.
- `biominer registry build|build-classification|audit`: production registry lifecycle.
- `biominer evaluation classify|review-queue`: offline evaluation and review preparation.
- `biominer storage doctor` and `biominer workstore doctor`: infrastructure checks.
- `biominer dev vision ...`: runtime checks, model prefetch, smoke tests, previews, evaluation, and benchmarks only.
- `biominer dev registry|flickr|comments ...`: bounded developer and operator utilities.

Developer benchmarks remain model-free where possible:

```bash
uv run biominer dev vision benchmark-plumbing --records 1000 --output-dir reports/vision_benchmarks/plumbing
uv run biominer dev vision benchmark-rolling-matrix --records 1000 --output-dir reports/vision_benchmarks/rolling
uv run biominer dev vision benchmark-cascade --output-dir reports/vision_benchmarks/cascade
```

## Verification

```bash
uv run ruff check .
.venv/bin/pytest -q
```

Tests use fake clients, fake classifiers, and synthetic images. They do not require live Flickr calls, CUDA, model weights, or downloaded photographs.

## Authoritative documentation

- [Registry and taxonomy](docs/registry.md)
- [Production workflow](docs/production.md)
- [Vision and classification](docs/vision.md)
- [Classification-v3 cutover](docs/migrations/classification-v3.md)

Repository invariants and agent implementation rules are defined in `AGENTS.md`.
