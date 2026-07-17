# Production workflow

`biominer run` is the only production pipeline entry point. Direct visual commands are intentionally absent.

## Stages

```text
resolve scope
  → build/validate registry
  → acquire, route and admit GBIF reference support
  → issue provisional or strict reference readiness
  → reuse/build reference embeddings and prototypes
  → compile queries
  → enqueue and poll Flickr metadata
  → detect and crop eligible butterflies
  → provisional reference ranking and/or seven-rank BioCLIP screening
  → join evidence
  → human-review Flickr release candidates
  → statistically audit reference-bank performance
  → review flagged references and selectively rerun affected evidence
  → final release gate
  → optional comment review
```

The default reference mode is `adaptive_gbif_fast_start`: automated reference
admission blocks first scoring, but reference human review does not. Flickr
human verification always blocks final occurrence release. Strict projects set
`--reference-admission-mode human_verified_strict`. The complete admission,
readiness, evidence-maturity and selective-rerun contract is documented in
[Adaptive GBIF fast-start](adaptive_gbif_fast_start.md).

The rolling worker uses bounded queues between staging, detection, crop materialization, image embedding, scoring, and commit. Workers return plain results; the main process merges, sorts, deduplicates, writes Parquet, registers immutable shards, and removes temporary images only after durable commit.

Cloud runs require S3-compatible storage and a PostgreSQL-compatible workstore. Local development can use filesystem storage and SQLite. Work claims, retry state, committed shard inventories, and source evidence make runs resumable and idempotent.

## Production cascade contract

The registry stores BioCLIP's supported identity ranks:
`KINGDOM → PHYLUM → CLASS → ORDER → FAMILY → GENUS → SPECIES`. Visual routing
uses the butterfly funnel: family top 1; genera within that family top 20 then
top 3; species beneath those genera top 20; distinct-prompt species top 5; and
final species top 1 selected from the top 5.

A genus score strictly above 0.90 activates the high-confidence shortcut and
routes species through only that top genus. At 0.90 or below, the broader genus
top-20 then top-3 route is retained. Every shortlist, score, margin, candidate
count, and routing mode is written to the classification output. Missing stored
ranks use semantic-parent proxies without making taxonomic assertions.

Every hierarchical run requires a validated `species_paths.parquet` in the
registry and a complete text-embedding cache built for the BioCLIP model ID and
checkpoint used by that run. Production verifies classification and prompt
versions, hierarchy and cache fingerprints, the exact staged-label set, and
embedding dimensions. Missing or mismatched taxonomy/cache input fails before
scoring; production has no direct-prompt fallback.

## Example

```bash
uv run biominer --config config/biominer.cloud.example.toml run \
  --taxon Papilionidae \
  --rank family \
  --registry-dir s3://biominer/registry/butterflies-v2 \
  --taxonomy-text-embedding-cache s3://biominer/cache/taxonomy/current/classification_text_embeddings.parquet \
  --output-prefix s3://biominer/runs/current \
  --classification-mode hierarchical_butterfly_classification \
  --reference-admission-mode adaptive_gbif_fast_start \
  --reference-source gbif \
  --initial-scoring-mode provisional_reference_ranking \
  --flickr-release-requires-human-review \
  --statistical-reference-audit
```

Use `storage doctor`, `workstore doctor`, and `run --dry-run` before a live run.
Dry-run resolves scope and records the plan and configured artifact paths; it
does not load or validate species paths or the embedding cache. Those
checks run, and fail closed, when the hierarchical vision stage initializes.

Cascade output persists `classification_fingerprint`,
`hierarchy_fingerprint`, and `embedding_cache_fingerprint`. Intermediate
`<rank>_top1` fields describe the rank-local raw-similarity winner, while
`selected_<rank>` fields describe the final reranked species-winning path; the
two are intentionally not interchangeable. Species audit arrays retain the
first-pass top 20, reranked top 5, and reported top 3. See
[Vision and classification](vision.md) for the complete field contract and the
registry migration notes before switching persisted artifacts or output roots.

## Durability and observability

Every stage reports command, run ID, PID, git SHA, inputs, outputs, timestamps, elapsed time, rows, bytes, retries, errors, and artifact paths. Unsupported metrics are `null` or `not_instrumented`. Long jobs write structured progress and checkpoints; repeated polling by operators is not part of the execution model.

Images, raw API dumps, models, caches, generated registry builds, large Parquet files, and secrets are runtime state and must not be committed.

Filesystem/SQLite local runs and S3/PostgreSQL cloud runs use the same semantic
artifact contracts. Mode and policy fingerprints, object checksums, immutable
readiness pins and checkpoint identities determine reuse; moving an artifact
between backends does not weaken validation or human-review gates.
