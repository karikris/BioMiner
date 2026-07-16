# Legacy-code audit — 2026-07-16

This audit combined repository import reachability, exact symbol searches,
Vulture, Pyflakes, full-suite coverage, Git history, CLI dispatch inspection,
and the architectural records under `docs/adr/` and `docs/schemas/`.

GitHits was used to verify the audit approach against the open-source
[`jendrikseipp/vulture`](https://github.com/jendrikseipp/vulture) and
[`nedbat/coveragepy`](https://github.com/nedbat/coveragepy) projects. The
important pattern was to treat static unused-code findings and missing coverage
as evidence requiring confirmation, not as proof by themselves. No external
implementation was copied.

## Findings and removals

- Removed the deprecated hidden `--taxonomy-candidate-table` compatibility
  option and duplicate manifest/report fields. `--registry-dir` is now the only
  taxonomy root accepted by production and developer commands.
- Removed the obsolete `require_cuda` boolean path. Device selection is now
  explicit through `device`.
- Removed the old `BIOMINER_RUNTIME_BASE_PATH` environment fallback.
- Removed a duplicated 204-line unified-taxonomy converter superseded by
  `UnifiedPathTaxonomyStore`.
- Removed uncalled query splitting, report-writing, translation-output,
  metadata-poller, workstore lifecycle, BioCLIP batch, bucket-view, and
  convenience APIs.
- Removed unreferenced private helpers and constants across registry, comments,
  taxonomy caching, evidence, storage, and test support.
- Removed historical tests whose only purpose was to assert that already
  deleted commands, modules, and aliases remained absent.

The test count changed from 2,334 to 2,302 because 32 historical tests were
removed. All remaining tests pass.

## Deliberately retained

The target-aware prompt pooling, fusion, full-frame, selective-decision, and
evaluation modules are not currently reachable from the main CLI import graph,
but they implement the normative future-state contracts documented in
`docs/schemas/target_aware_few_shot_contracts.md`. They were retained rather
than misclassified as legacy.

The diagnostic hierarchical/path-cascade implementations were also retained
because `docs/adr/target_aware_few_shot_classifier.md` explicitly requires
them as comparison baselines. Active legacy-checkpoint readers in reference
media download and evaluation were retained because they are still invoked to
upgrade durable artifacts safely. The optional `segment_anything` import is an
availability probe, not a dead import.

Run-artifact URI/path properties were retained as the declared future-state
artifact inventory even where individual stages are not wired yet.
