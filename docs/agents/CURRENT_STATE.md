# Current BioMiner state and active-goal safety

## Observed repository baseline

Updated from the local repository on 2026-07-18. Confirm Git and reports again
at the start of every task.

- Repository: `karikris/BioMiner`
- Default branch: `main`
- Latest verified implementation/release-gate commit:
  `477eaface3d1f5efa51255550f0ef8d6a7740f35`
- Active goal family: adaptive GBIF fast-start references
- Implementation phases: 0–13 complete, with the final self-identifying report
  resolved by the commit containing
  `reports/gbif_fast_start/final_report.json`.
- Current release-boundary evidence:
  - full regression: 2,531 passed in 109.26 seconds;
  - strict gate: 86 passed;
  - adaptive gate: 65 passed;
  - CLI gate: 143 passed;
  - pilot/fixture gate: 66 passed;
  - Ruff lint and locked dependency audit passed;
  - type and format checks retain explicit unconfigured failing baselines;
  - live source smoke and live Papilio scientific evaluation were not run.

This is an observed baseline, not a permanent assertion. Re-read local Git,
phase reports, and current tests before relying on these numbers.

GitHub cannot show uncommitted work from a running Codex session. The local
worktree is authoritative for the active task; the final report distinguishes
the ending implementation SHA from its self-referential containing commit.

## Active-goal protocol

Before touching a file:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git log -5 --oneline
git diff --stat
git diff --cached --stat
```

Then inspect:

```text
reports/gbif_fast_start/
provenance/githits.jsonl
docs/architecture/adaptive_gbif_reference_admission.md
the current goal/prompt or phase report
```

If a long process may be running, inspect the repository's workstore, PID,
lease, checkpoint, manifest, and structured logs. Do not infer process state
from silence.

### Never do during another active session

- `git reset`, `git restore`, `git checkout --`, `git clean`;
- branch switching without explicit goal authority;
- pull/rebase/merge on a dirty worktree;
- repository-wide format or import rewrites;
- staging all files blindly;
- amending or squashing another session's commits;
- deleting checkpoints, caches, workstore rows, reports, or output roots;
- starting duplicate acquisition, model, evaluation, or publication jobs.

If the requested task overlaps active files, stop and report the exact overlap
instead of attempting to merge incomplete work.

## Current production direction

`ProductionRunRequest` currently defaults to:

```text
storage backend                  s3
workstore backend                postgres
reference admission mode        adaptive_gbif_fast_start
reference source                gbif
initial scoring mode            provisional_reference_ranking
Flickr release human review     required
statistical reference audit     required
stages                          adaptive reference production stages
```

Three reference modes exist:

```text
adaptive_gbif_fast_start
human_verified_strict
human_verified_flagged_only
```

The adaptive stage plan is:

```text
resolve taxon scope
→ build registry
→ geographic spread
→ compile queries
→ enqueue and poll Flickr
→ Flickr geographic clustering
→ regional candidate generation
→ reference metadata and media
→ reference deduplication
→ reference quality routing
→ reference admission
→ reference embeddings
→ reference prototypes
→ provisional Flickr scoring
→ Flickr human verification
→ statistical reference audit
→ targeted reference review
→ affected reference rebuild
→ affected record rescore
→ final quality gate
```

Manual review stages may not be auto-completed.

## Accepted adaptive semantics

The accepted ADR is:

```text
docs/architecture/adaptive_gbif_reference_admission.md
```

It establishes:

- provisional, provider-asserted GBIF support as the default fast-start path;
- strict human-verified support as a compatibility/high-stakes path;
- targeted review and selective rerun for flagged species;
- mandatory human review before final Flickr inclusion;
- non-probabilistic provisional ranking;
- statistical auditing with human-reviewed Flickr labels;
- explicit evidence maturity and fail-closed release.

Key implementation locations:

```text
src/biominer/references/admission.py
src/biominer/references/admission_eligibility.py
src/biominer/references/admission_compiler.py
src/biominer/references/readiness.py
src/biominer/references/escalation.py
src/biominer/references/targeted_review.py
src/biominer/references/bank_revision.py

src/biominer/bioclip/reference_embeddings.py
src/biominer/bioclip/provisional_prototypes.py
src/biominer/bioclip/provisional_ranking.py
src/biominer/bioclip/reference_quality.py

src/biominer/run/stages.py
src/biominer/run/adaptive_config.py
src/biominer/run/support_dependencies.py
src/biominer/run/orchestrator.py

src/biominer/evaluation/
src/biominer/reports/
```

Verify exact filenames locally; the active goal may add or move modules.

## Legacy documentation warning

At the observed baseline:

- `README.md` still describes the old family-first funnel and 0.90 genus
  shortcut.
- `docs/production.md` still describes crop materialization and hierarchical
  production.
- `src/biominer/run/orchestrator.py` retains legacy cascade and visual-mode
  compatibility fields alongside adaptive defaults.
- `src/biominer/run/stages.py` retains legacy, strict reference-first, and
  adaptive stage plans.

Therefore:

- do not treat legacy README prose as active-goal authority;
- do not delete compatibility paths without an explicit migration task;
- do not let compatibility paths silently control adaptive output;
- update stale docs only when the active goal reaches its documentation phase
  or explicitly requests the change.

## Remaining live and human work

The implementation, migration/documentation, fixture-backed pilot, and release
verification phases are complete. This does not complete the scientific run.
The authoritative remaining-work ledger is in
`reports/gbif_fast_start/final_report.json` and currently includes:

- live current-policy GBIF acquisition and durable-media admission;
- live BioCLIP embedding, prototype, and Flickr scoring with instrumentation;
- 50 source-bound representative Flickr reviews;
- a sufficient-sample statistical audit;
- targeted reference review and selective rerun only if legitimately flagged.

Provider-asserted GBIF support remains provisional, current quality metrics are
unavailable, and no live production improvement or scientific release is
claimed.

## Repository map

```text
src/biominer/registry/          taxonomy, names, query compilation, spread
src/biominer/flickr_fetch/      planner, pollers, rate accounting, geography
src/biominer/geography/         cells, distance, validation
src/biominer/candidates/        regional and visual candidate unions
src/biominer/references/        acquisition, QA, admission, review, readiness
src/biominer/detection/         detector interfaces and routing policy
src/biominer/vision/            full-frame inputs, gates, rolling/cloud work
src/biominer/bioclip/           embeddings, prototypes, ranking, scoring
src/biominer/ml/                classifiers, calibration, non-match policy
src/biominer/evaluation/        labels, splits, metrics, sampling, reports
src/biominer/run/               stages, orchestration, paths, manifests
src/biominer/workstore/         SQLite/PostgreSQL work state and leases
src/biominer/storage/           local/cloud objects, Parquet, handoffs
src/biominer/reports/           structured stage and scientific reporting
tests/                          deterministic default suite
reports/gbif_fast_start/        active-goal evidence and phase reports
```
