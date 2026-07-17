# Adaptive GBIF reference workflow — release verification

The adaptive workflow's core release gates are verified from base commit
`c47a004bc2bcc11759df51fc8851d3b1b74788ee` plus the Task 13.4 report,
validator, provenance entry, and eight Ruff F401/F841 cleanups. The final full
regression passed 2,530 tests in 97.84 seconds, Ruff lint passed, the locked dependency graph has
no known vulnerabilities, generated artifacts parse cleanly, and the secret
review found no tracked private key or common live-token prefix.

This is deliberately not an “everything is green” claim. The repository does
not yet enforce a Ruff formatting profile or a mypy configuration. An
unconfigured format check would change 308 files, while mypy reports 892 errors
across 114 of 247 package files (94 errors across 21 of 38 changed source files).
Those are recorded toolchain gaps; this release task does not disguise them with
broad formatting changes or type suppressions.

## Test evidence

| Gate | Result |
|---|---:|
| Strict admission and readiness | 86 passed |
| Adaptive admission | 65 passed |
| CLI | 143 passed |
| Pilot, audit, review, rerun, and evaluation fixtures | 66 passed |
| Focused readiness/provisional/targeted/selective paths | 4 passed |
| GBIF source fixtures | 75 passed |
| Lint-cleanup regression | 208 passed |
| Full repository regression | 2,530 passed |

The optional live-source smoke test was not run because release-scoped provider
credentials and a durable live-media corpus were unavailable. The fixture gate
is a substitute only for interface and policy validation; no live network,
species-quality, or production-corpus conclusion is claimed.

## Scientific invariants

The passing strict, adaptive, CLI, pilot, readiness, review, reuse, rescore, and
holdout tests jointly verify that adaptive admission is the default while strict
mode remains usable; provider assertions are not called human verification;
unreviewed Flickr labels cannot reach final evaluation; unreviewed GBIF evidence
cannot reach calibration or final-test partitions; statistical audit is
mandatory before release conclusions; review and rescoring stay targeted;
unaffected embeddings are reused; raw margins are not probabilities; and
reference-derived artifacts bind admission mode, policy version, and policy
fingerprint.

## Supply-chain and artifact evidence

The lock resolves 44 packages under Python 3.14.5 and `uv` 0.11.19. Exporting
that locked graph into `pip-audit` found no known vulnerabilities (the local,
unpublished `biominer` package is necessarily skipped). Eighty-six JSON files,
three JSONL files, and one Parquet file all parse. No tracked machine-local path,
pending release placeholder, archive/cache candidate, or diff-whitespace error
was found.

`detect-secrets` reported 122 heuristic matches in 56 files: 98 semantic/content
hashes, 16 credential field names or placeholders, six fake basic-auth values,
and two fake Base64 test values. Manual category review plus a filename-only
scan for private-key blocks and common live-token prefixes found no deployable
secret. The machine-readable companion report preserves these counts and every
explicit limitation.
