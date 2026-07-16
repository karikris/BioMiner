# Classification-v3 cutover

Status: required persisted-artifact and output-schema migration.

Classification-v2 is not upgraded in place. V2 and v3 use some overlapping
artifact filenames but incompatible manifests, physical schemas, hierarchy
semantics, prompts, caches, work identities, and output rows. Preserve the old
registry, cache, work state, and committed outputs unchanged.

## Versioned roots

Use separate roots for each v3 artifact class:

- accepted GBIF spine, retained as read-only input: `data/registry/butterflies-v2`;
- classification-v3 overlay: `data/registry/classification-v3`;
- classification-v3 embedding cache: `data/cache/classification-v3`;
- classification-v3 run output: `data/runs/classification-v3`.

Do not write v3 into a directory containing a v2
`classification_manifest.json`. The writer checks the existing version before
writing and fails without replacing it.

## Rebuild the classification overlay

Build a v3 overlay from the accepted GBIF `taxa.parquet`:

```bash
uv run biominer registry build-classification \
  --registry-dir data/registry/butterflies-v2 \
  --output-dir data/registry/classification-v3 \
  --source-json config/taxonomy/papilionoidea_classification_v3.json
```

`--registry-dir` is the base-registry input. `--output-dir` is the new overlay
root supplied to production as `--registry-dir`.

A full fresh base-registry build is also valid:

```bash
uv run biominer registry build \
  --output-dir data/registry/butterflies-v3 \
  --registry-version butterflies-v3 \
  --workers 8 \
  --progress-every 100 \
  --checkpoint-every 500 \
  --max-retries 5
```

The full registry build compiles classification-v3 unless
`--skip-classification` is supplied.

## Build a new embedding cache

A v2 cache cannot be reused. Build from the new overlay with the BioCLIP model
and checkpoint that production will use:

```bash
uv run biominer dev vision build-text-embedding-cache \
  --registry-dir data/registry/classification-v3 \
  --output data/cache/classification-v3/classification_text_embeddings.parquet \
  --device auto \
  --batch-size 256
```

The loader validates the v3 manifest and artifact checksums, fatal-QA status,
exact staged prompt set, hierarchy fingerprint, model ID, model checkpoint,
embedding dimensions, normalized vectors, label hashes, and cache
self-fingerprint.

The two commands above consume and produce local paths. Build and validate
locally, then deploy the immutable overlay and cache through the project's
external artifact-publishing process. Do not pass S3 URIs to these builders.

## Expected identities

`classification_manifest.json` must contain:

- `classification_version`: `butterfly-classification-v3.0.0`;
- `prompt_version`: `butterfly-six-rank-prompts-v4`;
- `rank_order`: `FAMILY, SUBFAMILY, TRIBE, SUBTRIBE, GENUS, SPECIES`;
- `qa_status`: `passed` and `fatal_finding_count`: `0`;
- nonblank `classification_fingerprint` and `hierarchy_fingerprint`.

The embedding cache has no separate string-valued schema-version field. Its
exact physical schema and row-level classification, prompt, hierarchy, model,
checkpoint, label-hash, embedding, and `embedding_cache_fingerprint` fields are
validated together.

New cascade output uses:

- `classifier_schema_version`: `butterfly-cascade-output-v1.0.0`;
- `pruning_trace_version`: `global-rank-pruning-v1`.

The run manifest remains `run_manifest.json` with integer `schema_version: 1`;
its model configuration records the v3 inputs and fixed cascade settings.

## Dry-run the new paths

```bash
uv run biominer --config config/biominer.local.example.toml run \
  --taxon Papilionidae \
  --rank family \
  --registry-dir data/registry/butterflies-v2 \
  --registry-dir data/registry/classification-v3 \
  --taxonomy-text-embedding-cache data/cache/classification-v3/classification_text_embeddings.parquet \
  --output-prefix data/runs/classification-v3 \
  --storage-backend local \
  --workstore-backend sqlite \
  --classification-mode hierarchical_butterfly_classification \
  --dry-run
```

Dry-run checks parsing, configuration, base-registry scope resolution, and the
emitted plan. It records but does not open or validate the classification-v3
overlay or cache. Those inputs fail closed when the hierarchical vision stage
initializes.

## Output incompatibility

Classification-v2 object-score Parquet does not satisfy
`butterfly-cascade-output-v1.0.0`. It lacks the exact six-rank typed schema and
the global current-rank top-three, reviewed-SUBTRIBE-skip, rank-top1 versus
winning-path, and species `20 → 5 → 3` audit semantics.

Do not cast old columns, fill missing columns, relabel old rows as v3, append v3
rows to old datasets, or reinterpret committed v2 Parquet. Existing committed
parts remain immutable historical output. Analytical consumers must keep the
datasets separate or perform an explicit version-aware union that retains each
source schema identity.

## Work-key cutover

Work keys change intentionally. V3 computation identity includes the ordered
six ranks, fixed `global_rank_top_k` strategy and width 3, fixed species widths
20/5/3, classification and prompt versions, classification and hierarchy
fingerprints, embedding-cache fingerprint, and rerank-prompt identity.

Completed or pending v2 work is not completion evidence for v3. Enqueue new v3
work and write it to the new output root. Retry, claim, lease, and attempt
metadata do not change the immutable computation identity.

## Rollback

Before cutover, retain:

- the v2 registry and cache roots;
- the v2 output root and committed Parquet;
- the associated workstore records;
- the exact previous v2-capable release or Git SHA.

To roll back, stop v3 producers, deploy the previous v2-capable release, and
point it only at the original v2 registry, cache, work state, and output root.
The post-migration binary is not a v2 compatibility reader. Do not delete or
overwrite v3 artifacts during rollback; retain them for audit and a later
controlled retry.
