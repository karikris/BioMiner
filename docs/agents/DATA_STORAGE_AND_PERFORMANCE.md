# Data, storage, workers, and performance

## Data stack

Use:

```text
Polars       tabular transformation
Parquet      durable tabular artifacts
DuckDB       joins, analytics, summaries, QA
PyArrow      interoperability and bounded batch I/O
JSON         small configuration, manifests, checkpoints, reports
```

Do not add pandas or CSV workflows to production paths.

## Production and local backends

Production defaults:

```text
S3-compatible object storage
PostgreSQL workstore
```

Local development/test may explicitly use:

```text
filesystem artifact store
SQLite workstore
```

Do not make local storage behavior the implicit production contract.

## Artifact rules

Every durable artifact should have:

- schema version;
- semantic identity or fingerprint;
- physical checksum;
- input fingerprints;
- producer Git SHA;
- source/provider versions;
- row and byte counts;
- created/committed timestamps;
- explicit evidence maturity;
- manifest entry.

Rules:

1. Write immutable parts.
2. Sort deterministically.
3. Use Zstandard compression unless a measured case requires otherwise.
4. Commit data before publishing its manifest.
5. Write the manifest last.
6. Never delete source cache before the durable commit is verified.
7. Never overwrite a create-only publication silently.
8. Preserve failed publication audit evidence.
9. Treat a schema/policy/model change as an identity change.

## Semantic and physical identity

Do not conflate:

```text
semantic fingerprint
file SHA-256
source record hash
image content hash
duplicate-group identity
work-item ID
artifact URI
```

Use the repository's canonical semantic hash implementation. Do not invent
another serializer or hash preimage for an existing contract.

## Workstore and leases

Work items must be:

- idempotent;
- lease-safe;
- retryable;
- bounded;
- resumable;
- observable;
- versioned by semantic inputs.

A worker:

- claims bounded work;
- records worker ID and lease;
- renews or completes explicitly;
- does not publish duplicate parts;
- leaves retryable failure evidence;
- does not convert failure into success.

The coordinator/main process owns deterministic merge, sort, deduplication,
artifact commit, and publication.

## Concurrency

- Bound queues by items and bytes.
- Bound submitted futures; do not enqueue an entire corpus.
- Reuse pooled network clients.
- Honor provider retry and backoff contracts.
- Keep counters thread/process safe.
- Avoid CPU thread oversubscription from Python, BLAS, Polars, PyTorch, and
  model workers.
- Use one persistent model worker per accelerator by default.
- Add concurrent model processes only after measured throughput and memory
  tests.
- Workers return plain/immutable results.
- Main/coordinator writes artifacts.

## Model lifecycle and cache keys

Load YOLOE and BioCLIP once per persistent worker.

Cache embeddings using all semantic inputs, including:

```text
image content hash
visual-input kind and transform fingerprint
model ID and revision
preprocessing fingerprint
embedding dimension
route/domain contract
```

Do not recompute when only:

- review status changes;
- candidate union expands;
- a reference is excluded;
- map/report views change.

Reference-bank revision should produce an impact analysis before rebuilding.

## Selective recomputation

On reference revision:

1. Determine changed species/routes/scopes.
2. Reuse unchanged reference and Flickr embeddings.
3. Filter excluded references without vector recomputation.
4. Embed only newly admitted media.
5. Rebuild affected prototypes/indexes.
6. Refit only models/calibrators whose training evidence changed.
7. Rescore only records whose candidate evidence depends on changed species.
8. Report reused and recomputed work.

Do not launch a full rerun as the default remediation.

## Long-running jobs

A long job must record:

- run ID;
- PID or worker ID;
- branch and Git SHA;
- config and policy fingerprints;
- input and output URIs;
- stage;
- start time;
- progress;
- checkpoint;
- rate/retry state;
- errors;
- last durable commit.

Start once. Do not repeatedly poll through the agent. Use structured logs,
manifests, workstore state, and checkpoints.

## Metrics

Report when instrumented:

```text
rows/items in and out
distinct keys
duplicates removed
null/invalid keys
API calls and retries
429/rate-limit events
seconds and throughput
bytes read/written
partitions pruned
cache hits/misses
embeddings reused
model loads
peak RSS
accelerator memory
queue depth
checkpoint count
publication status
manual review count
selective-rerun ratio
```

Unsupported values are `null`, `unavailable`, or `not_instrumented`. Never
guess.

## Performance changes

Before optimizing:

1. Define the bottleneck.
2. Capture a reproducible baseline.
3. Preserve exact semantic outputs.
4. Measure time, memory, I/O, and work reuse.
5. Add a regression test with host-variance tolerance.
6. Report the test environment.

The current adaptive fixture baseline protects:

- time to first provisional score;
- zero reference review before first scoring;
- embedding reuse;
- selective rerun ratio;
- peak traced memory.

It is fixture evidence, not a live-corpus performance claim.

## Secrets and runtime data

Never commit:

- `.env`;
- provider/API keys;
- database DSNs;
- signed URLs;
- OAuth tokens;
- raw API dumps;
- downloaded Flickr/GBIF/iNaturalist images;
- model weights;
- local databases;
- caches;
- generated registry builds;
- large Parquet or DuckDB runtime outputs;
- temporary logs containing credentials.

Use environment variables or approved secret stores.
