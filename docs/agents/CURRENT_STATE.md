# Current BioMiner state and active-goal safety

## Observed repository baseline

Updated from the local repository on 2026-07-18. Confirm Git and reports again
at the start of every task.

- Repository: `karikris/BioMiner`
- Default branch: `main`
- Phase 0 dynamic-pooling design baseline:
  `299914548b407b439cd36d1aa99397b41aa827f1`
- Latest verified dynamic-pooling implementation commit:
  `cd37037a98a9239c2ff4bb5d30c661e9c950ce66`
  (`test(references): validate geographic reference index`)
- Active goal family: geography-conditioned dynamic global/local reference
  pooling
- Dynamic-pooling phases: Phase 0 baseline, audit and design complete; Phase 1
  cross-repository contract alignment complete; Phase 2 geographic reference
  indexing implementation complete through Task 2.1.4. The Task 2.1 gate and
  completion evidence are active; Phase 3 canonical Flickr work units are the
  next implementation boundary after that gate closes.
- Prior adaptive GBIF fast-start phases: 0–13 complete, with the final
  self-identifying report resolved by the commit containing
  `reports/gbif_fast_start/final_report.json`.
- Current release-boundary evidence:
  - dynamic-pooling Task 1.2 full regression: 2,660 passed in 100.67 seconds;
  - Task 1.2 focused cross-repository contracts: 33 passed in 0.17 seconds;
  - Task 2.1 final full regression: 2,730 passed in 101.37 seconds;
  - Task 2.1 reference/geography gate: 133 passed in 1.77 seconds;
  - Task 2.1 artifact round-trip: four Parquet artifacts and the JSON manifest
    passed semantic fingerprint and complete supplied physical-checksum checks;
  - dynamic-pooling Task 0.2 ADR/audit gate: 17 passed;
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
reports/geo_dynamic_pooling/
reports/gbif_fast_start/
provenance/githits.jsonl
provenance/task_pushes.jsonl
docs/architecture/geography_conditioned_dynamic_pooling.md
docs/architecture/statistical_support_and_human_verification.md
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

## Accepted dynamic-pooling semantics

Phase 0 accepted two additional ADRs:

```text
docs/architecture/geography_conditioned_dynamic_pooling.md
docs/architecture/statistical_support_and_human_verification.md
```

They establish:

- immutable cached embedding identity separate from versioned global/local
  comparison-plan membership;
- an explicit, diverse global safety pool and a local pool or exact
  local-unavailable reason;
- family evidence as a batching/retrieval accelerator, never a hard gate;
- geography as candidate/reference evidence, never identity proof or absence;
- deterministic, bounded uncertainty expansion that reuses embeddings;
- raw component scores and margins as non-probabilistic evidence;
- separate human-review, calibration, statistical-support, release-readiness
  and downstream-publication authorities; and
- immutable TaxaLens/ButterflyLens handoffs pinned to committed contracts.

Phase 0 is architecture evidence only. Reference indexes, pool/comparison-plan
schemas, hybrid candidate strategies, scoring, calibration, pilots and release
verification remain implementation work in Phases 1–16.

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

## Remaining implementation, live and human work

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

For the active dynamic-pooling goal, the next implementation boundary is Phase
1 contract alignment. No global/local pool artifact, strategy superiority,
live dynamic score, calibrated probability, statistical-support result or new
release-ready occurrence is claimed by the Phase 0 ADRs.

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
reports/geo_dynamic_pooling/    dynamic-pooling baselines and phase reports
```
