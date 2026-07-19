# Task 8.2 completion — measured dynamic-pooling efficiency

Status: completed and pushed to `origin/main` through
`f6be531e01d64be4e7e6f168313e66ecd3b25010`.

## Delivered

- A versioned embedding-reuse metric combines actual Flickr persistence
  request/hit/miss/materialization counters with validated reference-expansion
  cache identities. Flickr and reference roles remain separate, while a
  same-unit total is available.
- Family, candidate and pool matrix caches retain independent requests, hits,
  misses, materializations, rows, bytes and evictions. Worker-cache hits are
  not conflated with within-batch shared pool references or cross-batch reloads.
- A fingerprinted JSON/Markdown efficiency report combines those component
  metrics with a validated selective-rescore plan. Only `reuse_prior_score`
  rows count planned score-record avoidance; `selectively_rescore` rows remain
  planned work and are never called completed execution.
- Unobserved rates remain null and unavailable. Encoder seconds, score seconds,
  avoided bytes, cost and energy remain null and `not_instrumented`; no
  estimated or assumed savings fields are accepted.

## Measured gate

- Metrics and no-guess fixture suite: 91 passed in 3.03 seconds.
- Full regression: 2,921 passed in 106.61 seconds.
- Repository-wide Ruff lint and changed-file format checks: passed.
- Estimated/assumed savings field scan: zero matches.
- Provenance: 160 JSONL records; all four Task 8.2 records state
  `skipped_user_directive`, `solution_id: null`, and no GitHits call.
- Remote verification: `origin/main` resolved to `f6be531e…` after push.

The fixture observed seven embedding requests: five reuse events and two
materializations. Its worker caches observed seven matrix requests, four hits
and three materializations (six rows, 48 bytes); pool batching added three
within-batch shared references and recorded three cross-batch reloads. The
validated two-record rescore plan reused one prior score and planned one
selective rescore.

## Claim boundary

These are deterministic counter and plan fixtures, not a live execution or
benchmark. The one prior-score reuse is plan-derived avoidance if the plan is
executed, not a completed score receipt. Counts do not establish seconds,
bytes saved, cost, energy, throughput, scientific performance, taxonomy,
human review, occurrence release or deployment. Human-review sample planning
begins in Phase 9.

GitHits contributed no code or architecture because the user disabled all
further calls for this goal.
