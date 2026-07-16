# Phase 15 prototype integration go/no-go

Decision: **GO for explicit prototype integration only**.

This decision does not authorize a production-default change, scientific
release, public display of research-only reference images, or claims of
calibrated classification accuracy. Missing or contradictory required
evidence would produce `NO_GO`.

## Entry gates

| # | Gate | Result | Evidence summary | Limitation |
|---:|---|---|---|---|
| 1 | Prototype support bank frozen | PASS | 81-row bank `papilio-demoleus-prototype-bank-20260716` with frozen support hash | Prototype screening only |
| 2 | Bank marked `prototype_only` | PASS | Bank status and semantics explicitly prototype-only | None |
| 3 | Licences and attribution complete | PASS | 81/81 attribution, licence, and licence URI fields complete | 79 references are research-only |
| 4 | Exact duplicates resolved | PASS | Zero unresolved duplicate conflicts; 81 unique image hashes, media IDs, and observation IDs | Two records lack owner evidence |
| 5 | Adult, larval, specimen routes separate | PASS | Route mixing and cross-route neighbour checks pass | No pinned reference or larval support-train prototype |
| 6 | Frozen support embeddings exist | PASS | 81 finite, unit-normalized 1,024-dimensional embeddings | None |
| 7 | Target and competitor scoring works | PASS | Target once and 34 species candidates on all 13,496 classified records | Retrieval evidence, not accuracy |
| 8 | Target never hierarchy-pruned | PASS | No higher-rank pruning; target scored on every classified record | None |
| 9 | B0-B16 executable matrix ran | PASS | 81 records, zero skips, 19 experiment variants | B11/B12 transformed inputs unavailable |
| 10 | Staged Flickr classification ran | PASS | 13,496 of 13,501 classified with resumable checkpoint | Five retryable source failures |
| 11 | Model and data fingerprints exist | PASS | Bank, split, config, model, classifier, policy, and output fingerprints frozen | None |
| 12 | Limitations explicit | PASS | Policy and Phase 14 report contain limitations and review plan | None |
| 13 | Scores not called probabilities | PASS | Benchmark, staged, policy, and report semantics all prohibit probability interpretation | Policy remains uncalibrated |
| 14 | Human verification not falsely claimed | PASS | Human-verified count is zero throughout tracked evidence | Expert review still required |

All 14 required gates pass.

## Authorization boundary

Authorized:

- add and test an explicit Build Week target-aware prototype mode;
- require the frozen prototype readiness, support-bank, model, and policy
  identities;
- integrate the complete target and competitor scoring path;
- expose prototype status, raw margins, abstention, and limitations.

Not authorized:

- changing the current production default;
- silently falling back to a legacy or open classifier;
- weakening licence, attribution, duplicate, route, reconciliation, or
  fingerprint checks;
- calling raw scores probabilities;
- claiming scientific accuracy or human verification;
- publishing research-only image copies.

## Evidence and reproducibility

The machine-readable companion records the SHA-256 of every tracked input
manifest, the audited git SHA, all 14 gates, their evidence, limitations, and
the authorization scope. The underlying local Parquet audit also confirmed:

- 81/81 frozen rows have complete attribution, licence, and licence URI;
- 81 unique source-image hashes, media IDs, and observation IDs;
- 81 finite 1,024-dimensional frozen embeddings;
- 13,496/13,496 staged rows score the target;
- no staged row applies hierarchy pruning;
- every staged row scores 34 species candidates.

S3 was not used.

## Verification

- Phase 14 and Phase 15 focused contract suite: 26 passed.
- Full repository suite: 2,313 passed in 71.49 seconds.
- Ruff, JSON validation, CLI help, and `git diff --check`: passed.

## Next task

Task 15.1: add an explicit, selectable Build Week target-aware prototype
classification mode while leaving the production default unchanged.
