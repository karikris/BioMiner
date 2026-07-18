# Task 13.2 completion — dynamic-pooling CLI and configuration

Status: completed and pushed to `origin/main` through
`5166cf8408331898694cfd1ab5994c075a62458b`.

## Delivered

BioMiner now has one typed, fingerprinted dynamic-pooling settings artifact and
seven `biominer dynamic-pooling` commands: build the reference geography index,
plan pools, score pools, build review samples, audit quality, plan a selective
rerun, and export a handoff. The commands declare 22 exact input bindings and
15 intended output bindings across ten stage references.

Every command is plan-first. A dry run validates local, `file`, or `s3`
artifact bindings, emits a deterministic plan fingerprint, and may atomically
persist the JSON plan. Persisted plans are revalidated against their operation,
stages, exact inputs and outputs, settings selection state, fixed non-authority
fields, derived readiness fields, and fingerprint.

Strategy and fusion defaults remain explicitly unselected. Pool planning,
scoring, and handoff plans expose missing evidence-selected methods as readiness
issues rather than inventing defaults. Structural validity is kept distinct
from scientific readiness. Representative and targeted review outputs also
remain separate.

## Fail-closed boundary and verification

Production adapters are intentionally not connected in this task. Every
non-dry-run invocation fails before writing a plan or launching work. Every plan
states that it has no calibration, human-verification, statistical-support, or
occurrence-release authority.

- Focused configuration/CLI gate: 27 passed in 0.49 seconds.
- CLI suite: 137 passed in 11.29 seconds.
- Full regression: 3,123 passed in 115.05 seconds.
- Changed-file Ruff, format and `git diff --check`: passed.
- Remote `origin/main` resolved to `5166cf8…` after the implementation push.

No model inference, human review, calibration, quality audit, selective rerun,
handoff export, or release was performed. Raw scores remain distinct from
probabilities, and missing geography remains distinct from biological absence.
GitHits contributed no code or architecture because the user disabled all
further calls for this goal.
