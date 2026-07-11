# Production workflow

`biominer run` is the only production pipeline entry point. Direct visual commands are intentionally absent.

## Stages

```text
resolve scope
  → build/validate registry
  → compile queries
  → enqueue and poll Flickr metadata
  → detect and crop eligible butterflies
  → five-rank BioCLIP screening
  → join evidence
  → summarize and queue review
  → optional comment review
```

The rolling worker uses bounded queues between staging, detection, crop materialization, image embedding, scoring, and commit. Workers return plain results; the main process merges, sorts, deduplicates, writes Parquet, registers immutable shards, and removes temporary images only after durable commit.

Cloud runs require S3-compatible storage and a PostgreSQL-compatible workstore. Local development can use filesystem storage and SQLite. Work claims, retry state, committed shard inventories, and source evidence make runs resumable and idempotent.

## Example

```bash
uv run biominer --config config/biominer.cloud.example.toml run \
  --taxon Papilionoidea \
  --rank family \
  --registry-dir s3://biominer/registry/butterflies-v2 \
  --taxonomy-candidate-table s3://biominer/registry/butterflies-v2 \
  --taxonomy-text-embedding-cache s3://biominer/registry/butterflies-v2/classification_text_embeddings.parquet \
  --output-prefix s3://biominer/runs \
  --classification-mode hierarchical_butterfly_classification
```

Use `storage doctor`, `workstore doctor`, and `run --dry-run` before a live run.
The embedding cache is optional for diagnostic compatibility, but production
hierarchical runs should provide it; otherwise the manifest records
`direct_prompt_fallback` and BioCLIP repeatedly encodes rank prompts.

## Durability and observability

Every stage reports command, run ID, PID, git SHA, inputs, outputs, timestamps, elapsed time, rows, bytes, retries, errors, and artifact paths. Unsupported metrics are `null` or `not_instrumented`. Long jobs write structured progress and checkpoints; repeated polling by operators is not part of the execution model.

Images, raw API dumps, models, caches, generated registry builds, large Parquet files, and secrets are runtime state and must not be committed.
