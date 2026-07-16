# Phase 15 final verification

Result: **passed for the local-only Build Week prototype**.

Scientific release remains unauthorized. The production default remains
unchanged. S3 was not permitted or used.

## Command results

| Verification | Result |
|---|---|
| Focused prototype tests | 69 passed in 4.85s |
| Complete repository suite | 2,332 passed in 81.33s |
| Ruff lint | Passed repository-wide |
| Ruff format check | Passed for the Phase 15 Python additions |
| `ty` type check | Passed for the prototype evidence module |
| `git diff --check` | Passed |
| Dependency audit | No known vulnerabilities |
| Tracked-file secret scan | No committed credential; zero high-confidence findings |
| Five-image real MPS smoke | Passed; temporary output only |
| 100-image staged run | Passed |
| 1,000-image staged run | Passed |
| Full 13,501-record staged run | 13,496 classified; five retryable failures |
| Checkpoint resume | Passed without repeated stage/model work |
| Embedding cache resume | 81 reused, zero recomputed |
| CLI smoke checks | Five commands passed |
| Tracked artifact audit | Zero images, models, databases or cache paths |

The dependency audit initially found Pillow 12.2.0 advisories. The lock was
updated, within the existing `pillow>=12` constraint, to Pillow 12.3.0 and the
audit then reported no known vulnerabilities.

## Real execution evidence

The fresh five-image run loaded BioCLIP and YOLOE on MPS and passed with report
fingerprint
`sha256:f2088f252396496b3841f19267810a958409eff3a3b31915ff17046c6b8e3618`.
Its output was written under `/tmp` and is not committed.

The completed staged run resumed from SQLite without new work:

| Stage | Planned | Classified | Failures | Records/s | Peak RSS |
|---|---:|---:|---:|---:|---:|
| P1 | 100 | 100 | 0 | 2.012120 | 419,020,800 |
| P2 | 1,000 | 1,000 | 0 | 2.391706 | 445,857,792 |
| P3 | 13,501 | 13,496 | 5 | 2.274524 | 1,765,261,312 |

The five failures remain retryable download/decode failures and were not
converted into biological negatives.

## Security and repository hygiene

`detect-secrets` reported 65 heuristic matches in tracked files. They were
reviewed as hashes, pinned revisions, example placeholders, environment-variable
names, explicit test fixtures, and release-report terminology. A separate high-confidence scan for private
keys and common provider-token formats found zero matches.

Git tracks zero source images, model weights, databases, and cache paths.
The pre-existing untracked `.DS_Store` was left untouched and unstaged.

## GitHits

GitHits was invoked three times for Task 15.4. The service entered a repeated
grounding loop, then timed out, and a narrower repository search returned a
backend error. No external result was fabricated; the report structure retains
the accepted Task 15.3 pattern of paired JSON/Markdown, immutable evidence
identities, explicit failures, and limitations.

See `prototype_final_verification.json` for exact command strings and structured
results.
