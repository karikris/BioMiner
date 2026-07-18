# Task 15.2 completion — production-configuration decision

Status: completed and pushed to `origin/main` through
`30319fa7b7e9f24256556b293cf2e2db6e6ce2e7`.

## Complete selection ablation

The published table contains all 24 combinations of three candidate schedules,
two pool variants, and four raw fusion methods. Each row carries the same nine
selection criteria: target/candidate recall, reviewed precision and confidence
bounds, family/geographic subgroup behavior, review workload, computation,
embedding/matrix reuse, MPS memory, target-pruning regression, and unsupported
statistical claims.

Fixture structural recall, observed cached-vector work, reuse counts, and
target preservation remain separate from unavailable reviewed accuracy,
precision bounds, subgroup estimates, comparable timing, and MPS peak memory.
No missing measurement is encoded as zero performance. All rows remain
production-ineligible.

## Decision and unchanged defaults

The frozen policy returns `insufficient_evidence`, not rejection of measured
production performance. Reuse, target preservation, and the no-unsupported-
claim contract pass their software or fixture gates. Six criteria remain
blocking because evidence is ineligible, unavailable, or unmet.

Zero variants are eligible. Candidate strategy, pool variant, fusion method,
and selection evidence remain unset. Current and resulting runtime settings
share exact fingerprint
`sha256:0fd197b2650a79d99970cada3dcbabe9980c5a265d9d71f929bbcf6f51e13e7d`.
No runtime configuration changed.

## Integrated report and evidence boundary

The final pilot report binds the frozen plan, structural candidate results,
168 raw score projections, observed embedding/matrix reuse, separate
representative/targeted review work, complete selection table, and production
decision to exact semantic fingerprints.

The current execution is fixture-backed. Four historical real-execution
manifests remain a separate inventory and do not count as current results.
There are zero source-bound human labels and zero completed real reviews, so
reviewed precision and subgroup support remain unavailable and the effective
real-review shortfall remains 86 of 86. Raw scores are not probabilities,
missing geography is not biological absence, and no occurrence release is
authorized.

## Verification and provenance

- Complete pilot evaluation: 56 passed in 70.61 seconds.
- Full BioMiner regression: 3,225 passed in 164.55 seconds.
- Changed-file Ruff, formatting, and `git diff --check`: passed.
- Remote `origin/main` resolved to `30319fa…` after the implementation push.
- GitHits calls: zero, per the user's directive; direct and architectural
  contribution from GitHits: none.

This task completes the fixture pilot and production-decision implementation.
It does not select a default, prove improvement, complete human review, or
authorize scientific release.
