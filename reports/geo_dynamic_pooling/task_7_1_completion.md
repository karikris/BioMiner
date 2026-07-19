# Task 7.1 completion — raw component scoring

Status: completed and pushed to `origin/main` through
`ea85c00e321b21654c0934ee0f8783e468e76f78`.

## Delivered

- Complete family matrices produce one raw cosine, deterministic rank and
  next-row margin for every family; scores do not prune candidates.
- Every candidate has separate global prototype, nearest independent
  reference-observation and effective top-k mean cosines. Duplicate media from
  one observation are collapsed before observation-balanced scoring.
- Available local pools expose the same raw reference components plus their
  geographic scope. Unavailable local pools retain an exact reason, null local
  components and configured shortfall; global values are never substituted.
- Global/local comparison records expose local-minus-global signed deltas,
  absolute disagreement, component ranks, rank movement and top-one agreement.
  Positive rank movement means a candidate moved down in local ordering.
- Support, top-k and observation-independence coverage remain separate from
  score evidence. Query, model, matrix, candidate and pool identities are
  validated and all output rows and sets are semantically fingerprinted.

## Measured gate

- Focused numeric/matrix suite: 27 passed in 0.20 seconds.
- Adjacent dynamic-pooling suite: 269 passed in 19.74 seconds.
- Full regression: 2,876 passed in 104.89 seconds.
- Repository-wide Ruff: passed.
- Exact score-language scan: zero disallowed calibrated-score terms.
- Provenance: 148 valid JSONL records; all five Task 7.1 records state
  `skipped_user_directive`, `solution_id: null`, and no GitHits call.
- Remote verification: `origin/main` resolved to `ea85c00e...` after push.

The deterministic fixture retains one available and one unavailable local
candidate, verifies exact component arithmetic and coverage, and separately
forces a two-candidate global/local rank reversal. These are software-contract
results, not live BioCLIP or scientific-performance evidence.

## Claim boundary

Task 7.1 does not fuse or calibrate the raw components, select a strategy,
change a production default, verify taxonomy, authorize occurrence release or
complete human review. Versioned experimental fusion and candidate ranking
begin in Task 7.2.

GitHits contributed no code or architecture because the user disabled all
further calls for this goal.
