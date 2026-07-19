# Task 14.1 completion — TaxaLens evidence handoff

Status: completed and pushed to `origin/main` through
`1acc128dfdefb69150db0255b00258155ec1e358`.

## Delivered

BioMiner now publishes one complete, create-only TaxaLens dynamic-pool handoff.
Six canonical Parquet artifacts carry candidate scores, photo summaries, pool
plans, members, summaries, and candidate sets. A seventh Parquet artifact
projects only the selected representative probability sample, retaining the
validated sampling policy and register fingerprints, seed, strata, inclusion
probability, inverse-probability weight, independence groups, geographic
cluster, and explicit no-geo state.

An optional JSON sidecar projects only a validated grouped quality report. It
retains review counts, independence and effective-sample counts, per-metric
status, estimates, confidence intervals, and insufficiency reasons. A reviewed
but insufficient report is still an available artifact; its estimates remain
unavailable rather than becoming zero. Representative and targeted review,
raw scores and probabilities, and review and release remain separate.

Geographic-impact cells are explicitly unavailable because TaxaLens owns the
baseline-provider union and impact materialization. BioMiner does not fabricate
baseline counts or infer biological absence from missing geography.

## Publication and pinned compatibility

The publisher stages and validates all artifacts, writes the nine-role product
manifest last, builds a deterministic content-addressed archive, verifies its
embedded `storage-handoff-inventory-v1.0.0`, then atomically publishes the
directory. The archive contains nine files when reviewed quality is available
and eight when no reviewed quality report exists.

TaxaLens advances from the historical `c5e87ea…` pin to exact committed object
`e845dd9…`. Compatibility tests read that object through `git show`, never the
dirty sibling worktree, and execute its committed archive verifier against the
generated fixture. The consumer accepted the archive's member set, byte counts,
SHA-256 values, inventory path, and BioMiner source commit.

## Verification and authority boundary

- Pinned-SHA handoff/archive gate: 50 passed in 3.04 seconds.
- Full regression: 3,143 passed in 114.26 seconds.
- Changed-file Ruff, formatting, JSON parsing, and `git diff --check`: passed.
- Remote `origin/main` resolved to `1acc128…` after the implementation push.
- GitHits calls: zero, per the user's directive.

This handoff does not perform a live model run, production TaxaLens import,
database write, calibration, human review, or occurrence release. Every
artifact and the manifest remain non-authoritative for scientific release.
