# BioMiner agent instruction index

The root [`AGENTS.md`](../../AGENTS.md) contains only always-needed rules.
Load topic documents only when relevant.

| Document | Use |
|---|---|
| [`CURRENT_STATE.md`](CURRENT_STATE.md) | New session/goal; active-goal status; current architecture; stale-doc warning |
| [`GIT_AND_PROVENANCE.md`](GIT_AND_PROVENANCE.md) | Before branch, commit, push, merge, publication, or provenance work |
| [`TOOLS_AND_SKILLS.md`](TOOLS_AND_SKILLS.md) | GitHits, Valyu, MCP, skills, dependency research, large outputs |
| [`SCIENCE_AND_PIPELINE.md`](SCIENCE_AND_PIPELINE.md) | Scientific or pipeline behavior |
| [`DATA_STORAGE_AND_PERFORMANCE.md`](DATA_STORAGE_AND_PERFORMANCE.md) | Data formats, cloud/local storage, workers, checkpoints, optimization |
| [`TESTING_AND_RELEASE.md`](TESTING_AND_RELEASE.md) | Tests, live checks, phase/release verification |
| [`TASK_TEMPLATE.md`](TASK_TEMPLATE.md) | Task plan and completion report |

## Source-of-truth order

1. Explicit current goal.
2. Accepted ADR/versioned contract.
3. Current code and tests.
4. Commit-bound reports/manifests.
5. README/general docs.

`README.md` and `docs/production.md` now describe the adaptive full-frame
dynamic-pooling route as production direction. Their explicitly labelled
legacy compatibility sections do not override adaptive-reference contracts.

Update `CURRENT_STATE.md` at major goal/phase boundaries. Keep temporary phase
detail out of root `AGENTS.md`.
