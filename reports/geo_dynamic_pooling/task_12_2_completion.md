# Task 12.2 completion — verified selective reuse and rerun

Status: completed and pushed to `origin/main` through
`074e8671c7aac2e6a32705c8aa82b8b07c06b042`.

## Delivered

Reference vectors are now reused only when input content, producer/model and
preprocessing identities match a validated cache entry. Metadata-only bank
changes may reuse the vector; new or changed content is encoded, and removed
support is explicitly filtered. Flickr reuse is deduplicated by the exact
embedding fingerprint and remains independent of score invalidation.

The combined execution DAG orders embedding cache misses before affected
pools, affected matrices and affected individual scoring records. Every
required executor is preflighted before work starts. Reuse and exclusion rows
never invoke a materializer. Operations and execution receipts are
deterministic and tamper-evident.

## Measured fixture work

- Reference requirements: 5 (2 reused, 2 materialized, 1 excluded).
- Unique Flickr vectors: 2 (1 reused, 1 materialized) across 3 score records.
- Complete plan: 14 operations (7 executed, 6 reused, 1 excluded).
- Executed work: 2 reference vectors, 1 Flickr vector, 1 pool, 1 matrix and 2
  individual score records.
- Runtime savings: `not_instrumented`; no value was estimated.

## Gate and limits

- Selective rerun and adjacent cache/impact suite: 85 passed in 2.34 seconds.
- Full regression: 3,091 passed in 121.11 seconds.
- Changed-file Ruff, format and `git diff --check`: passed.
- Remote `origin/main` resolved to `074e867…` after the implementation push.

This is deterministic software-fixture evidence with fake executors. It did not
run live BioCLIP, rebuild a live pool, rescore live Flickr evidence, measure a
production speedup, select a fusion strategy, calibrate probabilities, perform
human review or authorize occurrence release. GitHits made no architecture or
code contribution because the user disabled every further call for this goal.
